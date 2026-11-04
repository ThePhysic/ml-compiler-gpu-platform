"""A deliberately small, typed, functional tensor SSA graph IR.

All values are materialized contiguous tensors. Layout changes are explicit
copy operations; no hidden views or aliasing enter lifetime analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any

import numpy as np

from .errors import GraphError

DTYPES = {"float16": np.dtype("float16"), "float32": np.dtype("float32")}
ELEMENTWISE = {"add", "mul", "relu", "gelu", "identity"}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"

    def __post_init__(self):
        if not isinstance(self.shape, (tuple, list)):
            raise GraphError("shape must be a list or tuple of positive integers")
        object.__setattr__(self, "shape", tuple(self.shape))
        if not self.shape or any(type(d) is not int or d <= 0 for d in self.shape):
            raise GraphError(f"shape must have positive static integer dimensions: {self.shape}")
        if not isinstance(self.dtype, str) or self.dtype not in DTYPES:
            raise GraphError(f"unsupported dtype {self.dtype!r}; supported: {tuple(DTYPES)}")

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPES[self.dtype].itemsize

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(math.prod(self.shape[i + 1:]) for i in range(len(self.shape)))

    def to_dict(self):
        return {"shape": list(self.shape), "dtype": self.dtype, "strides": list(self.strides), "layout": "contiguous"}


@dataclass
class Node:
    op: str
    inputs: tuple[str, ...]
    output: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.op, str) or not self.op:
            raise GraphError("operation name must be a nonempty string")
        if not isinstance(self.output, str) or not self.output:
            raise GraphError("node output must be a nonempty string")
        if not isinstance(self.inputs, (tuple, list)) or any(not isinstance(x, str) or not x for x in self.inputs):
            raise GraphError("node inputs must be a list or tuple of nonempty names")
        if not isinstance(self.attrs, dict) or any(not isinstance(x, str) for x in self.attrs):
            raise GraphError("operation attributes must be a dict with string keys")
        self.inputs = tuple(self.inputs)

    def to_dict(self):
        return {"op": self.op, "inputs": list(self.inputs), "output": self.output, "attrs": self.attrs}


def _broadcast(a, b):
    try:
        return tuple(np.broadcast_shapes(a, b))
    except ValueError as exc:
        raise GraphError(f"incompatible broadcast shapes {a} and {b}") from exc


def infer_node(node: Node, specs: dict[str, TensorSpec]) -> TensorSpec:
    """Check operator semantics and attributes, returning the output type."""
    counts = {"matmul": (2,), "add": (2,), "mul": (2,), "relu": (1,), "gelu": (1,),
              "identity": (1,), "softmax": (1,), "layernorm": (3,), "reshape": (1,),
              "transpose": (1,), "sum": (1,), "fused_matmul": (2, 3)}
    attrs = {"softmax": {"axis"}, "layernorm": {"eps"}, "reshape": {"shape"},
             "transpose": {"axes"}, "sum": {"axis", "keepdims"},
             "fused_matmul": {"activation"}, "fused_elementwise": {"steps"}}
    if node.op not in counts and node.op != "fused_elementwise":
        raise GraphError(f"unsupported operation {node.op!r} at {node.output}")
    if not isinstance(node.attrs, dict) or set(node.attrs) - attrs.get(node.op, set()):
        raise GraphError(f"unsupported attributes for {node.op}: {node.attrs}")
    if node.op in counts and len(node.inputs) not in counts[node.op]:
        raise GraphError(f"{node.op} expects {counts[node.op]} inputs, got {len(node.inputs)}")
    if not node.inputs or any(x not in specs for x in node.inputs):
        raise GraphError(f"missing input to {node.output}: {node.inputs}")
    inputs = [specs[x] for x in node.inputs]
    first = inputs[0]
    if any(s.dtype != first.dtype for s in inputs):
        raise GraphError(f"dtype mismatch at {node.output}; implicit promotion is unsupported")
    shape = first.shape
    if node.op in {"matmul", "fused_matmul"}:
        a, b = inputs[:2]
        if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
            raise GraphError(f"matmul requires (M,K) @ (K,N), got {a.shape}, {b.shape}")
        shape = (a.shape[0], b.shape[1])
        if len(inputs) == 3 and inputs[2].shape not in {(shape[1],), (1, shape[1])}:
            raise GraphError("fused matmul bias must have shape (N,) or (1,N)")
        activation = node.attrs.get("activation", "none")
        if not isinstance(activation, str) or activation not in {"none", "relu", "gelu"}:
            raise GraphError("unsupported fused matmul activation")
    elif node.op in {"add", "mul"}:
        shape = _broadcast(first.shape, inputs[1].shape)
    elif node.op == "softmax":
        axis = node.attrs.get("axis", -1)
        if type(axis) is not int or axis not in {-1, len(shape) - 1}:
            raise GraphError("softmax supports only the last axis")
    elif node.op == "layernorm":
        if any(s.shape != (shape[-1],) for s in inputs[1:]):
            raise GraphError("layernorm weight/bias must match last input dimension")
        eps = node.attrs.get("eps", 1e-5)
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise GraphError("layernorm eps must be finite and positive")
    elif node.op == "reshape":
        try:
            result = TensorSpec(tuple(node.attrs["shape"]), first.dtype)
        except (KeyError, TypeError) as exc:
            raise GraphError("reshape requires a positive static shape") from exc
        if result.numel != first.numel:
            raise GraphError("reshape changes element count")
        shape = result.shape
    elif node.op == "transpose":
        axes = node.attrs.get("axes", tuple(reversed(range(len(shape)))))
        if not isinstance(axes, (tuple, list)) or any(type(i) is not int for i in axes) or sorted(axes) != list(range(len(shape))):
            raise GraphError("transpose axes must be a permutation")
        shape = tuple(shape[i] for i in axes)
    elif node.op == "sum":
        axis = node.attrs.get("axis", -1)
        keep = node.attrs.get("keepdims", True)
        if type(axis) is not int or not -len(shape) <= axis < len(shape) or type(keep) is not bool:
            raise GraphError("sum requires valid integer axis and boolean keepdims")
        axis %= len(shape)
        shape = tuple(1 if i == axis else d for i, d in enumerate(shape)) if keep else shape[:axis] + shape[axis + 1:]
    elif node.op == "fused_elementwise":
        steps = node.attrs.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
            raise GraphError("fused elementwise region must have 1..16 steps")
        local = {name: specs[name] for name in node.inputs}
        produced = []
        for step in steps:
            if not isinstance(step, dict) or not {"op", "inputs", "output"} <= set(step) or set(step) - {"op", "inputs", "output", "attrs"}:
                raise GraphError("malformed fused elementwise step")
            sub = Node(step["op"], step["inputs"], step["output"], step.get("attrs", {}))
            if sub.op not in ELEMENTWISE or sub.output in local:
                raise GraphError("fused elementwise region must be elementwise SSA")
            local[sub.output] = infer_node(sub, local)
            produced.append(sub.output)
        if produced[-1] != node.output or any(local[x] != local[produced[-1]] for x in produced):
            raise GraphError("fused elementwise steps must share the final shape and dtype")
        return local[produced[-1]]
    return TensorSpec(shape, first.dtype)


@dataclass
class Graph:
    inputs: dict[str, TensorSpec]
    nodes: list[Node]
    outputs: tuple[str, ...]
    constants: dict[str, np.ndarray] = field(default_factory=dict)
    name: str = "graph"

    def __post_init__(self):
        if not isinstance(self.inputs, dict) or not isinstance(self.constants, dict) or not isinstance(self.nodes, (tuple, list)):
            raise GraphError("graph requires input/constant mappings and a node sequence")
        if not isinstance(self.outputs, (tuple, list)) or any(not isinstance(x, str) or not x for x in self.outputs):
            raise GraphError("graph outputs must be a list or tuple of nonempty names")
        self.nodes = list(self.nodes)
        self.outputs = tuple(self.outputs)

    def ordered_nodes(self) -> list[Node]:
        sources = set(self.inputs) | set(self.constants)
        if set(self.inputs) & set(self.constants):
            raise GraphError("input and constant names overlap")
        names = set(sources)
        for node in self.nodes:
            if not isinstance(node, Node) or not isinstance(node.output, str) or not node.output or node.output in names:
                raise GraphError(f"duplicate or invalid SSA value {getattr(node, 'output', None)!r}")
            if any(not isinstance(x, str) or not x for x in node.inputs):
                raise GraphError("node input names must be nonempty strings")
            names.add(node.output)
        if any(not isinstance(x, str) or not x for x in sources):
            raise GraphError("source names must be nonempty strings")
        if not self.outputs or len(set(self.outputs)) != len(self.outputs) or any(x not in names for x in self.outputs):
            raise GraphError("graph outputs must be nonempty, unique, and defined")
        for node in self.nodes:
            if any(x not in names for x in node.inputs):
                raise GraphError(f"undefined input at {node.output}: {node.inputs}")
        pending = list(self.nodes)
        ordered = []
        while pending:
            ready = [n for n in pending if set(n.inputs) <= sources]
            if not ready:
                raise GraphError("graph contains a dependency cycle")
            for node in ready:
                ordered.append(node)
                sources.add(node.output)
                pending.remove(node)
        return ordered

    def specs(self) -> dict[str, TensorSpec]:
        specs = dict(self.inputs)
        if any(not isinstance(s, TensorSpec) for s in specs.values()):
            raise GraphError("inputs require TensorSpec values")
        for name, array in self.constants.items():
            if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
                raise GraphError(f"constant {name} must be a contiguous NumPy tensor")
            specs[name] = TensorSpec(array.shape, array.dtype.name)
        for node in self.ordered_nodes():
            specs[node.output] = infer_node(node, specs)
        return specs

    def validate(self):
        self.specs()
        return self

    def to_dict(self):
        specs = self.specs()
        return {"name": self.name, "inputs": {k: s.to_dict() for k, s in self.inputs.items()},
                "constants": {k: {"spec": specs[k].to_dict(), "values": v.tolist()} for k, v in self.constants.items()},
                "nodes": [n.to_dict() for n in self.ordered_nodes()], "outputs": list(self.outputs),
                "values": {k: s.to_dict() for k, s in specs.items()}}

    def format(self):
        return json.dumps(self.to_dict(), indent=2)
