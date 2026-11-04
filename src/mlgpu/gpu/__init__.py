"""Lazy, explicitly selected Triton backend for the static tensor IR.

Importing this module does not import torch, Triton, or initialize CUDA.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
import sys

from ..errors import BackendUnavailable, ExecutionError, ResourceError

MAX_REDUCTION_WIDTH = 16384
MAX_REGION_STEPS = 16
MAX_REGION_INPUTS = 32
MAX_RANK = 8
_OPCODES = {"add": 0, "mul": 1, "relu": 2, "gelu": 3, "identity": 4}


@lru_cache(maxsize=1)
def _packages():
    if sys.platform != "linux":
        raise BackendUnavailable("Triton backend requires Linux; use the documented WSL2 environment on Windows")
    try:
        import torch
        import triton
    except ImportError as exc:
        raise BackendUnavailable("Triton backend requires optional torch and triton dependencies") from exc
    return torch, triton


def ensure_available():
    """Fail explicitly when the optional CUDA/Triton backend cannot execute."""
    torch, _ = _packages()
    if not torch.cuda.is_available():
        raise BackendUnavailable("Triton backend requires a CUDA-capable NVIDIA GPU and a working driver")
    if torch.version.hip is not None:
        raise BackendUnavailable("This backend targets NVIDIA CUDA; HIP is not supported")
    if torch.cuda.get_device_capability()[0] < 7:
        raise BackendUnavailable("This backend requires NVIDIA compute capability >= 7.0")


def _strides(shape):
    result = []
    stride = 1
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


def _region(node):
    steps = node.attrs.get("steps") if node.op == "fused_elementwise" else [
        {"op": node.op, "inputs": node.inputs, "output": node.output, "attrs": node.attrs}
    ]
    if not isinstance(steps, (tuple, list)) or not 1 <= len(steps) <= MAX_REGION_STEPS:
        raise ResourceError(f"GPU elementwise region requires 1..{MAX_REGION_STEPS} steps")
    if len(node.inputs) > MAX_REGION_INPUTS:
        raise ResourceError(f"GPU elementwise region supports at most {MAX_REGION_INPUTS} external inputs")
    bindings = {name: index for index, name in enumerate(node.inputs)}
    opcodes, arg0, arg1 = [], [], []
    for step in steps:
        op = step["op"]
        if op not in _OPCODES:
            raise ExecutionError(f"GPU elementwise region does not support {op!r}")
        args = step["inputs"]
        expected = 2 if op in ("add", "mul") else 1
        if len(args) != expected or any(name not in bindings for name in args):
            raise ExecutionError("GPU elementwise region has invalid SSA operands")
        opcodes.append(_OPCODES[op])
        arg0.append(bindings[args[0]])
        arg1.append(bindings[args[-1]])
        bindings[step["output"]] = len(node.inputs) + len(opcodes) - 1
    if steps[-1]["output"] != node.output:
        raise ExecutionError("GPU elementwise region's final value must be its output")
    return tuple(opcodes), tuple(arg0), tuple(arg1)


def _validate(node, inputs, output, torch):
    from ..ir import Graph, TensorSpec

    if len(inputs) != len(node.inputs):
        raise ExecutionError(f"GPU {node.op}: expected {len(node.inputs)} inputs, received {len(inputs)}")
    tensors = [*inputs, output]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ExecutionError("GPU inputs and output must be torch tensors")
    if not all(tensor.is_cuda for tensor in tensors):
        raise ExecutionError("GPU inputs and output must be CUDA tensors")
    if any(tensor.device != output.device for tensor in tensors):
        raise ExecutionError("GPU inputs and output must be on the same CUDA device")
    if any(tensor.dtype != output.dtype for tensor in tensors) or output.dtype not in (torch.float16, torch.float32):
        raise ExecutionError("GPU tensors require identical float16 or float32 dtype")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ExecutionError("GPU tensors must be contiguous; transpose operations are materialized explicitly")
    if any(tensor.ndim > MAX_RANK for tensor in tensors):
        raise ResourceError(f"GPU backend supports tensor rank <= {MAX_RANK}")
    if any(tensor.numel() >= 2**31 for tensor in tensors):
        raise ResourceError("GPU backend uses 32-bit indexing and requires fewer than 2^31 elements")
    seen = {}
    for name, tensor in zip(node.inputs, inputs):
        descriptor = _descriptor(tensor)
        if name in seen and seen[name] != descriptor:
            raise ExecutionError("Repeated GPU input names must refer to the same tensor storage and metadata")
        seen[name] = descriptor
    dtype = str(output.dtype).split(".")[-1]
    graph = Graph(
        {name: TensorSpec(tuple(tensor.shape), dtype) for name, tensor in zip(node.inputs, inputs)},
        [node], (node.output,),
    )
    spec = graph.specs()[node.output]
    if tuple(output.shape) != tuple(spec.shape):
        raise ExecutionError(f"GPU {node.op}: output shape {tuple(output.shape)} does not match {spec.shape}")
    start, end = output.data_ptr(), output.data_ptr() + output.numel() * output.element_size()
    for tensor in inputs:
        lo, hi = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
        if start < hi and lo < end:
            raise ExecutionError("GPU output must not overlap live input storage")


def _tile_config(config, inputs, torch):
    config = dict(config or {})
    unknown = set(config) - {"block_m", "block_n", "block_k", "num_warps"}
    if unknown:
        raise ExecutionError(f"Unknown GPU matmul configuration keys: {sorted(unknown)}")
    result = {"block_m": 32, "block_n": 64, "block_k": 32, "num_warps": 4}
    result.update(config)
    for key in ("block_m", "block_n"):
        if type(result[key]) is not int or result[key] not in (16, 32, 64, 128):
            raise ResourceError(f"GPU {key} must be one of 16, 32, 64, 128")
    if type(result["block_k"]) is not int or result["block_k"] not in (16, 32, 64):
        raise ResourceError("GPU block_k must be 16, 32, or 64")
    if type(result["num_warps"]) is not int or result["num_warps"] not in (4, 8):
        raise ResourceError("GPU num_warps must be 4 or 8")
    props = torch.cuda.get_device_properties(inputs[0].device)
    shared = 2 * result["block_k"] * (result["block_m"] + result["block_n"]) * inputs[0].element_size()
    capacity = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    if shared > capacity:
        raise ResourceError(f"GPU tile's conservative shared-memory estimate {shared} exceeds {capacity} bytes")
    return result


def _descriptor(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device, tensor.data_ptr())


def prepare_launch(node, inputs, output, config=None):
    """Validate and specialize a dispatch, returning a zero-argument callable.

    The callback captures immutable op/shape/configuration metadata and holds
    the tensor objects alive. Data may change between invocations; shape,
    dtype, strides, device, and storage address must remain stable. The cheap
    descriptor guard catches resize_/set_ before stale pointers can execute.
    """
    ensure_available()
    torch, triton = _packages()
    node = deepcopy(node)
    inputs = tuple(inputs)
    _validate(node, inputs, output, torch)
    from . import kernels

    common = {"enable_fp_fusion": False}
    op = node.op
    with torch.cuda.device(output.device):
        if op in _OPCODES or op == "fused_elementwise":
            codes, lhs, rhs = _region(node)
            shape = tuple(output.shape)
            padded = tuple((1,) * (len(shape) - tensor.ndim) + tuple(tensor.shape) for tensor in inputs)
            strides = tuple(_strides(s) for s in padded)
            kernel = kernels.elementwise
            grid = (triton.cdiv(output.numel(), 256),)
            args = (inputs, output, output.numel(), shape, _strides(shape),
                    padded, strides, codes, lhs, rhs, 256)
            options = {"num_warps": 4, **common}
        elif op == "reshape":
            kernel = kernels.copy_flat
            grid = (triton.cdiv(output.numel(), 256),)
            args = (inputs[0], output, output.numel(), 256)
            options = {"num_warps": 4, **common}
        elif op == "transpose":
            axes = tuple(node.attrs.get("axes", reversed(range(inputs[0].ndim))))
            source = _strides(tuple(inputs[0].shape))
            permuted = tuple(source[axis] for axis in axes)
            shape = tuple(output.shape)
            kernel = kernels.transpose_copy
            grid = (triton.cdiv(output.numel(), 256),)
            args = (inputs[0], output, output.numel(), shape, _strides(shape), permuted, 256)
            options = {"num_warps": 4, **common}
        elif op in ("softmax", "layernorm", "sum"):
            if op == "sum":
                axis = node.attrs.get("axis", -1) % inputs[0].ndim
                width = inputs[0].shape[axis]
                inner = math.prod(inputs[0].shape[axis + 1:])
                rows = output.numel()
            else:
                width = inputs[0].shape[-1]
                rows = inputs[0].numel() // width
            if width > MAX_REDUCTION_WIDTH:
                raise ResourceError(f"GPU {op} reduction width {width} exceeds supported maximum {MAX_REDUCTION_WIDTH}")
            block = triton.next_power_of_2(width)
            warps = 4 if block <= 2048 else 8
            grid = (rows,)
            options = {"num_warps": warps, **common}
            if op == "softmax":
                kernel = kernels.softmax_rows
                args = (inputs[0], output, width, block)
            elif op == "layernorm":
                kernel = kernels.layernorm_rows
                args = (*inputs, output, width, node.attrs.get("eps", 1e-5), block)
            else:
                kernel = kernels.sum_axis
                args = (inputs[0], output, width, inner, block)
        elif op in ("matmul", "fused_matmul"):
            tiles = _tile_config(config, inputs, torch)
            activation = node.attrs.get("activation", "none") if op == "fused_matmul" else "none"
            if activation not in ("none", "relu", "gelu"):
                raise ExecutionError(f"Unsupported GPU matmul activation {activation!r}")
            m, k = inputs[0].shape
            _, n = inputs[1].shape
            bias = inputs[2] if len(inputs) == 3 else output
            grid = (triton.cdiv(m, tiles["block_m"]), triton.cdiv(n, tiles["block_n"]))
            kernel = kernels.matmul_epilogue
            args = (
                inputs[0], inputs[1], bias, output, m, n, k, len(inputs) == 3,
                ("none", "relu", "gelu").index(activation),
                tiles["block_m"], tiles["block_n"], tiles["block_k"],
            )
            options = {"num_warps": tiles["num_warps"], "num_stages": 2, **common}
        else:
            raise ExecutionError(f"GPU backend has no lowering for operation {op!r}")

    tensors = (*inputs, output)
    descriptors = tuple(_descriptor(tensor) for tensor in tensors)
    device = output.device
    from triton.runtime.errors import OutOfResources

    def dispatch():
        if any(_descriptor(tensor) != desc for tensor, desc in zip(tensors, descriptors)):
            raise ExecutionError("Prepared GPU tensor metadata or storage changed; bind the execution again")
        try:
            with torch.cuda.device(device):
                compiled = kernel[grid](*args, **options)
                if not dispatch.kernel_metadata:
                    metadata = getattr(compiled, "metadata", None)
                    dispatch.kernel_metadata.update({
                        "kernel": kernel.fn.__name__,
                        "registers_per_thread": getattr(compiled, "n_regs", None),
                        "spill_slots": getattr(compiled, "n_spills", None),
                        "shared_memory_bytes": getattr(metadata, "shared", None),
                        "num_warps": options["num_warps"],
                        "grid": list(grid),
                    })
        except OutOfResources as exc:
            raise ResourceError(f"GPU kernel {op} exceeded device resources: {exc}") from exc

    dispatch.kernel_metadata = {}
    return dispatch


def launch(node, inputs, output, config=None):
    """Validate and write a caller-owned output using custom Triton.

    Returns None. Execution is asynchronous on the current CUDA stream.
    Repeated dispatch should use prepare_launch to move graph validation and
    indexing/configuration specialization out of the hot path.
    """
    prepare_launch(node, inputs, output, config)()
