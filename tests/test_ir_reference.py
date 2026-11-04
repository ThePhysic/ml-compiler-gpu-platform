import numpy as np
import pytest

from mlgpu.errors import GraphError, ExecutionError
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.reference import execute


def test_topological_order_independent_of_input_list_and_multiple_outputs():
    g = Graph({"x": TensorSpec((2, 3))}, [Node("relu", ("a",), "y"), Node("add", ("x", "x"), "a")], ("x", "y"))
    x = np.array([[-1, 2, 3], [4, -5, 6]], np.float32)
    assert [n.output for n in g.ordered_nodes()] == ["a", "y"]
    out = execute(g, {"x": x})
    np.testing.assert_array_equal(out["y"], np.maximum(x * 2, 0))
    assert not np.shares_memory(out["x"], x)


@pytest.mark.parametrize("shape,dtype", [((0, 2), "float32"), ((True, 2), "float32"), ((), "float32"), ((3,), "int32")])
def test_reject_bad_tensor_specs(shape, dtype):
    with pytest.raises(GraphError):
        TensorSpec(shape, dtype)


@pytest.mark.parametrize("nodes,outputs", [
    ([Node("relu", ("missing",), "y")], ("y",)),
    ([Node("relu", ("b",), "a"), Node("relu", ("a",), "b")], ("a",)),
    ([Node("relu", ("x",), "x")], ("x",)),
    ([Node("sin", ("x",), "y")], ("y",)),
    ([Node("relu", ("x",), "y", {"typo": 1})], ("y",)),
    ([], ("missing",)),
])
def test_reject_invalid_graphs(nodes, outputs):
    with pytest.raises(GraphError):
        Graph({"x": TensorSpec((2, 3))}, nodes, outputs).validate()


def test_broadcast_layout_reduce_and_reshape_semantics():
    g = Graph({"x": TensorSpec((2, 3)), "b": TensorSpec((3,))}, [
        Node("add", ("x", "b"), "a"), Node("transpose", ("a",), "t", {"axes": [1, 0]}),
        Node("sum", ("t",), "s", {"axis": 1, "keepdims": False}),
        Node("reshape", ("s",), "r", {"shape": [1, 3]})], ("t", "r"))
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.array([1, 3, 5], np.float32)
    result = execute(g, {"x": x, "b": b})
    np.testing.assert_array_equal(result["r"], (x + b).sum(0).reshape(1, 3))
    assert result["t"].flags.c_contiguous


def test_stable_softmax_and_constant_layernorm():
    g = Graph({"x": TensorSpec((2, 5)), "w": TensorSpec((5,)), "b": TensorSpec((5,))}, [
        Node("softmax", ("x",), "s"), Node("layernorm", ("x", "w", "b"), "ln")], ("s", "ln"))
    result = execute(g, {"x": np.full((2, 5), 10000, np.float32), "w": np.ones(5, np.float32), "b": np.zeros(5, np.float32)})
    np.testing.assert_allclose(result["s"], .2)
    np.testing.assert_array_equal(result["ln"], 0)


def test_reject_input_dtype_shape_and_layout():
    g = Graph({"x": TensorSpec((2, 3))}, [], ("x",))
    for values in ({}, {"x": np.zeros((2, 3), np.float16)}, {"x": np.zeros((3, 2), np.float32)}, {"x": np.zeros((3, 2), np.float32).T}):
        with pytest.raises(ExecutionError):
            execute(g, values)


def test_reject_bad_broadcast_matmul_reduction_and_norm_attrs():
    for node in [Node("add", ("x", "b"), "y"), Node("matmul", ("x", "b"), "y"),
                 Node("sum", ("x",), "y", {"axis": 3}), Node("softmax", ("x",), "y", {"axis": 0}),
                 Node("layernorm", ("x", "w", "w"), "y", {"eps": float("nan")})]:
        with pytest.raises(GraphError):
            Graph({"x": TensorSpec((2, 3)), "b": TensorSpec((4,)), "w": TensorSpec((3,))}, [node], ("y",)).validate()
