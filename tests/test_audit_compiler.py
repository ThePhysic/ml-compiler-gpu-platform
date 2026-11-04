"""Independent audit regressions for fusion overlap and inspection snapshots."""
from copy import deepcopy

import numpy as np
import pytest

from mlgpu.compiler import compile_graph
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.reference import execute
from mlgpu.runtime import Program


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("reverse_heads", [False, True])
@pytest.mark.parametrize("activation", [None, "relu", "gelu"])
def test_row_matmul_merge_fuses_each_epilogue_at_most_once(dtype, reverse_heads, activation):
    def spec(shape):
        return TensorSpec(shape, dtype)

    heads = [Node("matmul", ("x", "w"), "a"), Node("matmul", ("z", "v"), "b")]
    if reverse_heads:
        heads.reverse()
    nodes = heads + [Node("add", ("a", "b"), "sum")]
    output = "sum"
    if activation:
        nodes.append(Node(activation, ("sum",), "y"))
        output = "y"
    graph = Graph({"x": spec((1, 3)), "w": spec((3, 4)), "z": spec((1, 3)), "v": spec((3, 4))}, nodes, (output,))
    rng = np.random.default_rng(71)
    values = {name: rng.normal(0, .3, s.shape).astype(dtype) for name, s in graph.inputs.items()}
    original = deepcopy(graph.to_dict())
    expected = execute(graph, values)
    plan = compile_graph(graph)
    plan.graph.validate()
    assert len(plan.nodes) == 2
    assert {node.op for node in plan.nodes} == {"matmul", "fused_matmul"}
    np.testing.assert_array_equal(Program(graph).run(values)[output], expected[output])
    np.testing.assert_array_equal(execute(compile_graph(plan.graph).graph, values)[output], expected[output])
    assert graph.to_dict() == original


def fused_alias_graph():
    return Graph({"x": TensorSpec((3,))}, [
        Node("identity", ("x",), "a"),
        Node("fused_elementwise", ("a",), "y", {"steps": [
            {"op": "relu", "inputs": ["a"], "output": "y"}]})], ("y",))


def test_pass_snapshots_preserve_original_nested_region_operands():
    graph = fused_alias_graph()
    original = deepcopy(graph.to_dict())
    plan = compile_graph(graph)
    folding = next(item for item in plan.passes if item["pass"] == "constant_folding")
    simplify = next(item for item in plan.passes if item["pass"] == "simplification")
    assert folding["after"]["nodes"][-1]["attrs"]["steps"][0]["inputs"] == ["a"]
    assert simplify["before"]["nodes"][-1]["attrs"]["steps"][0]["inputs"] == ["a"]
    assert simplify["before"]["nodes"][-1]["inputs"] == ["a"]
    assert simplify["after"]["nodes"][-1]["attrs"]["steps"][0]["inputs"] == ["x"]
    assert simplify["after"]["nodes"][-1]["inputs"] == ["x"]
    assert graph.to_dict() == original


def test_serialized_nested_attrs_cannot_mutate_source_graph_or_pass_history():
    graph = fused_alias_graph()
    original = deepcopy(graph.to_dict())
    serialized = graph.to_dict()
    serialized["nodes"][-1]["attrs"]["steps"][0]["inputs"][0] = "tampered"
    assert graph.to_dict() == original
    plan = compile_graph(graph)
    saved = deepcopy(plan.passes)
    plan.graph.nodes[-1].attrs["steps"][0]["inputs"][0] = "tampered"
    assert plan.passes == saved
