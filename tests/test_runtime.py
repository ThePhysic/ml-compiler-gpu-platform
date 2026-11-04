import numpy as np
import pytest

from mlgpu.errors import BackendUnavailable, ExecutionError, ResourceError
from mlgpu.examples import make_inputs, workload
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.reference import execute
from mlgpu.runtime import Program


@pytest.mark.parametrize("name", ["epilogue", "norm", "mlp", "softmax", "elementwise"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_full_graph_reference_runtime(name, dtype):
    graph = workload(name, 7, 19, 31, dtype)
    inputs = make_inputs(graph, seed=4)
    expected = execute(graph, inputs)
    for optimize in (False, True):
        program = Program(graph, optimize=optimize)
        bound = program.bind(inputs)
        actual = bound.run()
        for output in graph.outputs:
            np.testing.assert_array_equal(actual[output], expected[output])
        for name, allocation in program.plan.memory.allocations.items():
            assert np.shares_memory(bound.values[name], bound.arena)
            assert bound.values[name].nbytes == allocation.size


def test_constant_only_and_external_outputs():
    g = Graph({"x": TensorSpec((2,))}, [Node("add", ("c", "c"), "y")], ("x", "y", "c"), {"c": np.array([2, 4], np.float32)})
    out = Program(g).run({"x": np.ones(2, np.float32)})
    np.testing.assert_array_equal(out["y"], [4, 8])
    np.testing.assert_array_equal(out["c"], [2, 4])


def test_results_retained_across_repeated_runs_and_independent_bindings():
    g = workload("elementwise", 3, 7, 9)
    a, b = make_inputs(g, 1), make_inputs(g, 2)
    program = Program(g)
    first, second = program.bind(a), program.bind(b)
    retained = first.run()["y"]
    a["x"].fill(10)
    updated = first.run()["y"]
    assert not np.array_equal(retained, updated)
    np.testing.assert_array_equal(second.run()["y"], execute(g, b)["y"])
    assert not np.shares_memory(first.arena, second.arena)


def test_no_implicit_backend_fallback_and_memory_budget():
    with pytest.raises(BackendUnavailable):
        Program(workload(), backend="unknown")
    with pytest.raises(ResourceError):
        Program(workload(), memory_budget=1)


def test_runtime_validates_inputs_and_reentrancy():
    g = workload("elementwise")
    program = Program(g)
    with pytest.raises(ExecutionError):
        program.bind({})
    bound = program.bind(make_inputs(g))
    bound._lock.acquire()
    try:
        with pytest.raises(ExecutionError, match="reentrant"):
            bound.run()
    finally:
        bound._lock.release()
