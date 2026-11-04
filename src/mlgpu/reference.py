"""Numerical semantics: FP32 arithmetic, explicit tensor dtype boundaries."""
from __future__ import annotations

import numpy as np

from .errors import ExecutionError
from .ir import Graph, Node


def gelu(x):
    x = np.asarray(x, dtype=np.float32)
    return 0.5 * x * (1 + np.tanh(np.float32(np.sqrt(2 / np.pi)) * (x + np.float32(0.044715) * x * x * x)))


def evaluate_node(node: Node, values: dict[str, np.ndarray]) -> np.ndarray:
    raw = [values[name] for name in node.inputs]
    a = [v.astype(np.float32, copy=False) for v in raw]
    op, attrs = node.op, node.attrs
    dtype = raw[0].dtype
    if op == "fused_elementwise":
        local = {k: values[k] for k in node.inputs}
        for step in attrs["steps"]:
            sub = Node(step["op"], tuple(step["inputs"]), step["output"], step.get("attrs", {}))
            local[sub.output] = evaluate_node(sub, local)
        return local[node.output]
    if op in {"matmul", "fused_matmul"}:
        result = (a[0] @ a[1]).astype(dtype)
        if op == "fused_matmul":
            if len(a) == 3:
                result = (result.astype(np.float32) + a[2]).astype(dtype)
            activation = attrs.get("activation", "none")
            if activation == "relu":
                result = np.maximum(result.astype(np.float32), 0)
            elif activation == "gelu":
                result = gelu(result)
    elif op == "add":
        result = a[0] + a[1]
    elif op == "mul":
        result = a[0] * a[1]
    elif op == "relu":
        result = np.maximum(a[0], 0)
    elif op == "gelu":
        result = gelu(a[0])
    elif op == "softmax":
        exp = np.exp(a[0] - np.max(a[0], axis=-1, keepdims=True))
        result = exp / np.sum(exp, axis=-1, keepdims=True)
    elif op == "layernorm":
        centered = a[0] - np.mean(a[0], axis=-1, keepdims=True)
        variance = np.mean(centered * centered, axis=-1, keepdims=True)
        result = centered / np.sqrt(variance + attrs.get("eps", 1e-5)) * a[1] + a[2]
    elif op == "sum":
        result = np.sum(a[0], axis=attrs.get("axis", -1), keepdims=attrs.get("keepdims", True), dtype=np.float32)
    elif op in {"identity", "reshape"}:
        result = raw[0].reshape(attrs.get("shape", raw[0].shape))
    elif op == "transpose":
        result = np.transpose(raw[0], attrs.get("axes"))
    else:
        raise ExecutionError(f"reference: unsupported operation {op}")
    return np.ascontiguousarray(result, dtype=dtype)


def validate_inputs(graph: Graph, inputs: dict[str, np.ndarray]):
    if set(inputs) != set(graph.inputs):
        raise ExecutionError(f"input names: expected {sorted(graph.inputs)}, got {sorted(inputs)}")
    for name, spec in graph.inputs.items():
        a = inputs[name]
        if not isinstance(a, np.ndarray) or a.shape != spec.shape or a.dtype.name != spec.dtype or not a.flags.c_contiguous:
            raise ExecutionError(f"input {name}: requires contiguous {spec.dtype}{spec.shape}")


def execute(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    graph.validate()
    validate_inputs(graph, inputs)
    values = {**graph.constants, **inputs}
    for node in graph.ordered_nodes():
        try:
            values[node.output] = evaluate_node(node, values)
        except Exception as exc:
            raise ExecutionError(f"reference operation {node.op} -> {node.output}: {exc}") from exc
    return {name: values[name].copy() for name in graph.outputs}
