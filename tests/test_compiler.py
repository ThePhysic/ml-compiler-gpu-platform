"""Semantic, scheduling and alias safety checks for the original compiler."""
from copy import deepcopy
import json

import numpy as np
import pytest

from mlgpu.compiler import compile_graph, plan_memory, schedule_nodes
from mlgpu.errors import CompileError, ResourceError
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.reference import execute


def spec(shape=(4, 8), dtype="float32"):
    return TensorSpec(shape, dtype)


def epilogue(dtype="float32", activation="gelu"):
    return Graph({"a": spec((5, 7), dtype), "b": spec((7, 9), dtype), "bias": spec((9,), dtype)}, [
        Node("matmul", ("a", "b"), "mm"),
        Node("add", ("mm", "bias"), "biased"),
        Node(activation, ("biased",), "y"),
    ], ("y",))


def random_inputs(graph):
    rng = np.random.default_rng(79)
    return {name: rng.normal(size=s.shape).astype(s.dtype) for name, s in graph.inputs.items()}


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_matmul_bias_activation_fusion_preserves_values(dtype, activation):
    graph = epilogue(dtype, activation)
    original = deepcopy(graph.to_dict())
    values = random_inputs(graph)
    plan = compile_graph(graph)
    assert len(plan.nodes) == 1
    assert plan.nodes[0].op == "fused_matmul"
    assert plan.nodes[0].attrs["activation"] == activation
    np.testing.assert_array_equal(execute(graph, values)["y"], execute(plan.graph, values)["y"])
    assert graph.to_dict() == original
    assert "y" in plan.tiles
    assert plan.memory.peak_bytes < compile_graph(graph, optimize=False).memory.peak_bytes
    json.loads(plan.format())


def test_graph_output_prevents_erasing_materialization():
    graph = epilogue()
    graph.outputs = ("mm", "y")
    plan = compile_graph(graph)
    assert any(n.op == "matmul" and n.output == "mm" for n in plan.nodes)
    assert plan.memory.allocations["mm"].end == len(plan.nodes)
    assert set(execute(plan.graph, random_inputs(graph))) == {"mm", "y"}


def test_fanout_blocks_epilogue_fusion():
    graph = epilogue()
    graph.nodes.append(Node("mul", ("mm", "mm"), "other"))
    graph.outputs = ("y", "other")
    plan = compile_graph(graph)
    assert any(n.op == "matmul" for n in plan.nodes)
    values = random_inputs(graph)
    for name, value in execute(graph, values).items():
        np.testing.assert_array_equal(value, execute(plan.graph, values)[name])


def test_full_matrix_bias_is_not_epilogue_fused():
    graph = epilogue()
    graph.inputs["bias"] = spec((5, 9))
    assert all(n.op != "fused_matmul" for n in compile_graph(graph).nodes)


def test_activation_without_bias_fuses():
    graph = Graph({"a": spec((3, 7)), "b": spec((7, 11))}, [
        Node("matmul", ("a", "b"), "mm"), Node("relu", ("mm",), "y")], ("y",))
    plan = compile_graph(graph)
    assert plan.nodes[0].op == "fused_matmul"
    assert plan.nodes[0].inputs == ("a", "b")


def test_constant_fold_and_signature_preservation():
    graph = Graph({"unused_input": spec((2,))}, [
        Node("add", ("a", "b"), "s"), Node("relu", ("s",), "y")], ("y",),
        {"a": np.array([-3, 2], dtype=np.float32), "b": np.array([1, 5], dtype=np.float32)})
    plan = compile_graph(graph)
    assert plan.nodes == []
    assert set(plan.graph.inputs) == {"unused_input"}
    assert set(plan.graph.constants) == {"y"}
    assert plan.memory.peak_bytes == 0
    np.testing.assert_array_equal(plan.graph.constants["y"], [0, 7])


def test_external_outputs_have_no_arena_reservation():
    graph = Graph({"x": spec((8,))}, [], ("x", "c"), {"c": np.ones((3,), dtype=np.float32)})
    plan = compile_graph(graph, memory_budget=0)
    assert plan.memory.allocations == {}
    assert set(plan.memory.external_values) == {"x", "c"}
    assert set(execute(plan.graph, random_inputs(graph))) == {"x", "c"}


def test_internal_copies_simplified_but_public_output_names_preserved():
    graph = Graph({"x": spec((4,))}, [Node("identity", ("x",), "copy"),
        Node("reshape", ("copy",), "view", {"shape": (4,)}),
        Node("relu", ("view",), "r"), Node("identity", ("r",), "y")], ("y",))
    plan = compile_graph(graph)
    assert [n.output for n in plan.nodes] == ["r", "y"]
    assert plan.nodes[0].inputs == ("x",)
    assert execute(plan.graph, random_inputs(graph)).keys() == {"y"}


def test_dead_code_elimination_does_not_drop_input_signature():
    graph = Graph({"x": spec(), "z": spec()}, [Node("relu", ("x",), "y"), Node("relu", ("z",), "dead")], ("y",))
    plan = compile_graph(graph)
    assert [n.output for n in plan.nodes] == ["y"]
    assert set(plan.graph.inputs) == {"x", "z"}


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_elementwise_fusion_broadcast_and_fp16_rounding(dtype):
    graph = Graph({"x": spec((13, 7), dtype), "bias": spec((7,), dtype), "scale": spec((1,), dtype)}, [
        Node("add", ("x", "bias"), "a"), Node("mul", ("a", "scale"), "b"),
        Node("gelu", ("b",), "y")], ("y",))
    plan = compile_graph(graph)
    assert len(plan.nodes) == 1
    assert plan.nodes[0].op == "fused_elementwise"
    assert plan.nodes[0].inputs == ("x", "bias", "scale")
    values = random_inputs(graph)
    np.testing.assert_array_equal(execute(graph, values)["y"], execute(plan.graph, values)["y"])


def test_broadcast_changing_intermediate_shape_blocks_chain_fusion():
    graph = Graph({"x": spec((7,)), "z": spec((3, 7))}, [
        Node("relu", ("x",), "a"), Node("add", ("a", "z"), "b"), Node("relu", ("b",), "y")], ("y",))
    plan = compile_graph(graph)
    assert len(plan.nodes) == 2
    assert plan.nodes[0].op == "relu"
    values = random_inputs(graph)
    np.testing.assert_array_equal(execute(graph, values)["y"], execute(plan.graph, values)["y"])


def test_elementwise_fusion_step_bound():
    nodes = []
    current = "x"
    for i in range(37):
        output = f"v{i}"
        nodes.append(Node("relu", (current,), output))
        current = output
    plan = compile_graph(Graph({"x": spec()}, nodes, (current,)))
    assert len(plan.nodes) == 3
    assert all(len(n.attrs["steps"]) <= 16 for n in plan.nodes)


def test_dependency_schedule_reduces_branch_memory():
    graph = Graph({"x": spec((4096,))}, [
        Node("relu", ("x",), "a"), Node("relu", ("x",), "b"),
        Node("sum", ("a",), "a_small", {"axis": 0, "keepdims": True}),
        Node("sum", ("b",), "b_small", {"axis": 0, "keepdims": True}),
        Node("add", ("a_small", "b_small"), "y")], ("y",))
    order = schedule_nodes(graph)
    assert [n.output for n in order][:2] == ["a", "a_small"]
    optimized = plan_memory(graph, order)
    baseline = plan_memory(graph)
    assert optimized.peak_bytes < baseline.peak_bytes
    assert order == schedule_nodes(graph)


def test_memory_lifetimes_are_inclusive_and_outputs_pinned():
    graph = Graph({"x": spec((7,))}, [
        Node("relu", ("x",), "a"), Node("relu", ("a",), "b"),
        Node("relu", ("b",), "c"), Node("add", ("a", "c"), "y")], ("a", "y"))
    plan = compile_graph(graph, optimize=False)
    allocations = plan.memory.allocations
    assert allocations["a"].end == len(plan.nodes)
    for left_name, left in allocations.items():
        assert left.offset % 256 == 0
        assert left.size == graph.specs()[left_name].nbytes
        for right_name, right in allocations.items():
            if left_name == right_name:
                continue
            concurrent = max(left.start, right.start) <= min(left.end, right.end)
            overlapping = max(left.offset, right.offset) < min(left.offset + left.reserved_size, right.offset + right.reserved_size)
            assert not (concurrent and overlapping), (left_name, right_name)


def test_dead_buffers_reused_after_last_use_but_not_during_it():
    graph = Graph({"x": spec((17,))}, [Node("relu", ("x",), "a"),
        Node("relu", ("a",), "b"), Node("relu", ("b",), "y")], ("y",))
    memory = compile_graph(graph, optimize=False).memory
    assert memory.allocations["a"].offset != memory.allocations["b"].offset
    assert memory.allocations["a"].offset == memory.allocations["y"].offset
    assert memory.peak_bytes == 512
    assert memory.naive_bytes == 768


def test_best_fit_splits_and_coalesces_free_ranges():
    graph = Graph({"x": spec((1024,))}, [
        Node("relu", ("x",), "a"),
        Node("sum", ("a",), "s", {"axis": 0, "keepdims": True}),
        Node("relu", ("s",), "t"), Node("add", ("x", "t"), "big")], ("big",))
    memory = compile_graph(graph, optimize=False).memory
    # The first large range is split for t, then re-coalesced when allocating big.
    assert memory.peak_bytes <= memory.naive_bytes
    assert memory.allocations["big"].size == 4096


@pytest.mark.parametrize("budget", [-1, True, 1.5, "1024"])
def test_invalid_memory_budget_rejected(budget):
    with pytest.raises(CompileError, match="memory_budget"):
        compile_graph(epilogue(), memory_budget=budget)


def test_memory_limit_is_enforced_on_actual_aligned_arena():
    plan = compile_graph(epilogue())
    compile_graph(epilogue(), memory_budget=plan.memory.peak_bytes)
    with pytest.raises(ResourceError, match="arena requires"):
        compile_graph(epilogue(), memory_budget=plan.memory.peak_bytes - 1)


def test_unordered_input_nodes_compiled_topologically():
    graph = epilogue()
    graph.nodes.reverse()
    assert compile_graph(graph).nodes[0].op == "fused_matmul"


def test_unoptimized_plan_preserves_nodes_and_records_no_rewrite_passes():
    graph = epilogue()
    plan = compile_graph(graph, optimize=False)
    assert [n.op for n in plan.nodes] == ["matmul", "add", "gelu"]
    assert all(p["pass"] in {"dependency_scheduling", "memory_planning"} for p in plan.passes)
