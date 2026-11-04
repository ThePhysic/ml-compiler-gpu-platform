"""Compile once, bind independent arenas, and dispatch the explicit plan."""
from __future__ import annotations

from collections.abc import Mapping
import threading
import time

import numpy as np

from .errors import BackendUnavailable, ExecutionError, ResourceError
from .ir import DTYPES
from .reference import evaluate_node, validate_inputs


def _numpy_descriptor(array):
    return (tuple(array.shape), tuple(array.strides), array.dtype,
            array.__array_interface__["data"][0])


class Program:
    def __init__(self, graph, backend="reference", optimize=True, memory_budget=None, device="cuda:0"):
        from .compiler import compile_graph
        if backend not in {"reference", "gpu"}:
            raise BackendUnavailable(f"runtime backend {backend!r} is unsupported")
        start = time.perf_counter()
        self.plan = compile_graph(graph, optimize=optimize, memory_budget=memory_budget)
        self.compile_ms = (time.perf_counter() - start) * 1000
        self.backend = backend
        self.device = device
        if backend == "gpu":
            from .gpu import ensure_available
            ensure_available()

    def bind(self, inputs):
        """Validate inputs and allocate a separate execution arena."""
        return BoundExecution(self, inputs)

    def run(self, inputs):
        """Convenience execution, including validation/allocation and output copies."""
        return self.bind(inputs).run(copy_outputs=True)


class BoundExecution:
    """A non-reentrant execution instance; different bindings have separate arenas.

    run(copy_outputs=False) returns borrowed outputs valid until this instance is
    run again. GPU runs enqueue work on the current PyTorch CUDA stream; callers
    synchronize when observing timing or errors. Do not mutate inputs in flight.
    """
    def __init__(self, program, inputs):
        self.program = program
        self.plan = program.plan
        self.backend = program.backend
        self._lock = threading.Lock()
        graph = self.plan.graph
        specs = graph.specs()
        if self.backend == "reference":
            validate_inputs(graph, inputs)
            self.values = {**{name: value.copy() for name, value in graph.constants.items()}, **inputs}
            self._source_descriptor = _numpy_descriptor
            try:
                self.arena = np.empty(self.plan.memory.peak_bytes, dtype=np.uint8)
                for name, allocation in self.plan.memory.allocations.items():
                    spec = specs[name]
                    self.values[name] = np.ndarray(spec.shape, dtype=DTYPES[spec.dtype], buffer=self.arena, offset=allocation.offset)
            except (MemoryError, ValueError) as exc:
                raise ResourceError(f"runtime arena allocation failed: {exc}") from exc
        else:
            import torch
            from .gpu import prepare_launch, _descriptor
            self._source_descriptor = _descriptor
            device = torch.device(program.device)
            if device.type != "cuda":
                raise BackendUnavailable("GPU runtime requires a CUDA device")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if not isinstance(inputs, Mapping):
                raise ExecutionError("runtime inputs must be a mapping from tensor names to tensors")
            if set(inputs) != set(graph.inputs):
                raise ExecutionError(f"input names: expected {sorted(graph.inputs)}, got {sorted(repr(name) for name in inputs)}")
            for name, spec in graph.inputs.items():
                a = inputs[name]
                if not isinstance(a, torch.Tensor) or tuple(a.shape) != spec.shape or a.dtype != getattr(torch, spec.dtype) or a.device != device or not a.is_contiguous():
                    raise ExecutionError(f"input {name}: requires contiguous {spec.dtype}{spec.shape} on {device}")
                if a.is_neg() or a.is_conj():
                    raise ExecutionError(f"input {name}: GPU logical negative/conjugate views are unsupported; resolve tensor flags before binding")
            self.values = dict(inputs)
            try:
                with torch.cuda.device(device):
                    self._cuda_stream = torch.cuda.current_stream(device).cuda_stream
                    self._torch = torch
                    self._device = device
                    self.values.update({name: torch.from_numpy(a.copy()).to(device) for name, a in graph.constants.items()})
                    self.arena = torch.empty(self.plan.memory.peak_bytes, dtype=torch.uint8, device=device)
                    for name, allocation in self.plan.memory.allocations.items():
                        spec = specs[name]
                        self.values[name] = self.arena.narrow(0, allocation.offset, spec.nbytes).view(getattr(torch, spec.dtype)).reshape(spec.shape)
            except torch.OutOfMemoryError as exc:
                raise ResourceError(f"CUDA runtime arena requires {self.plan.memory.peak_bytes} bytes") from exc
            self._dispatches = [prepare_launch(node, [self.values[x] for x in node.inputs], self.values[node.output], self.plan.tiles.get(node.output)) for node in self.plan.nodes]
        self._source_descriptors = {name: self._source_descriptor(self.values[name])
                                    for name in (*graph.inputs, *graph.constants)}

    def run(self, copy_outputs=True):
        if not self._lock.acquire(blocking=False):
            raise ExecutionError("bound execution is not reentrant; create separate bindings")
        try:
            if self.backend == "gpu" and self._torch.cuda.current_stream(self._device).cuda_stream != self._cuda_stream:
                raise ExecutionError("bound execution belongs to its binding CUDA stream; bind separately on another stream")
            for name, expected in self._source_descriptors.items():
                try:
                    current = self._source_descriptor(self.values[name])
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise ExecutionError(f"bound source {name} changed; bind the execution again") from exc
                if current != expected:
                    raise ExecutionError(f"bound source {name} metadata or storage changed; bind the execution again")
            for index, node in enumerate(self.plan.nodes):
                try:
                    if self.backend == "reference":
                        np.copyto(self.values[node.output], evaluate_node(node, self.values))
                    else:
                        self._dispatches[index]()
                except Exception as exc:
                    if isinstance(exc, (BackendUnavailable, ResourceError)):
                        raise
                    raise ExecutionError(f"{self.backend} execution: {node.op} -> {node.output}: {exc}") from exc
            outputs = {name: self.values[name] for name in self.plan.graph.outputs}
            if copy_outputs:
                return {name: a.copy() if self.backend == "reference" else a.clone() for name, a in outputs.items()}
            return outputs
        finally:
            self._lock.release()
