"""Bounded workloads shared by demos, integration tests and benchmarks."""
import numpy as np

from .ir import Graph, Node, TensorSpec


def workload(name="epilogue", m=32, k=64, n=96, dtype="float32"):
    def spec(shape):
        return TensorSpec(tuple(shape), dtype)
    if name == "epilogue":
        graph = Graph({"x": spec((m, k)), "w": spec((k, n)), "b": spec((n,))}, [
            Node("matmul", ("x", "w"), "mm"), Node("add", ("mm", "b"), "biased"),
            Node("gelu", ("biased",), "y")], ("y",), name=name)
    elif name == "mlp":
        graph = Graph({"x": spec((m, k)), "w1": spec((k, n)), "b1": spec((n,)),
                       "w2": spec((n, k)), "b2": spec((k,))}, [
            Node("matmul", ("x", "w1"), "h0"), Node("add", ("h0", "b1"), "h1"),
            Node("gelu", ("h1",), "h2"), Node("matmul", ("h2", "w2"), "h3"),
            Node("add", ("h3", "b2"), "h4"), Node("add", ("h4", "x"), "y")], ("y",), name=name)
    elif name == "norm":
        graph = Graph({"x": spec((m, n)), "weight": spec((n,)), "bias": spec((n,)), "r": spec((m, n))}, [
            Node("layernorm", ("x", "weight", "bias"), "ln", {"eps": 1e-5}),
            Node("add", ("ln", "r"), "sum"), Node("gelu", ("sum",), "y")], ("y",), name=name)
    elif name == "softmax":
        graph = Graph({"x": spec((m, n)), "b": spec((n,))}, [
            Node("add", ("x", "b"), "logits"), Node("softmax", ("logits",), "y")], ("y",), name=name)
    elif name == "elementwise":
        graph = Graph({"x": spec((m, n)), "b": spec((n,)), "scale": spec((n,))}, [
            Node("add", ("x", "b"), "a"), Node("mul", ("a", "scale"), "b0"),
            Node("gelu", ("b0",), "y")], ("y",), name=name)
    elif name == "matmul":
        graph = Graph({"x": spec((m, k)), "w": spec((k, n))}, [Node("matmul", ("x", "w"), "y")], ("y",), name=name)
    else:
        raise ValueError(f"unknown workload {name!r}")
    return graph.validate()


def make_inputs(graph, seed=0):
    rng = np.random.default_rng(seed)
    result = {}
    for name, spec in graph.inputs.items():
        values = rng.normal(0, 0.3, spec.shape)
        if name == "weight":
            values += 1
        result[name] = np.ascontiguousarray(values, dtype=spec.dtype)
    return result
