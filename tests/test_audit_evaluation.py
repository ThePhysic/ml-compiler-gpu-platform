"""Additional audit regressions for numerical validation and input bounds."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlgpu.benchmark import compare_arrays
from mlgpu.errors import ExecutionError, GraphError
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.reference import execute
from mlgpu.runtime import Program


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_benchmark_rejects_matching_nonfinite_reference_and_output(value):
    values = {"y": np.array([value], np.float32)}
    with pytest.raises(ExecutionError, match="non-finite"):
        compare_arrays(values, values, "float32")


@pytest.mark.parametrize("width", [1, 3, 513, 2049])
def test_reference_large_constant_layernorm_rows_remain_zero(width):
    graph = Graph({"x": TensorSpec((2, width)), "w": TensorSpec((width,)), "b": TensorSpec((width,))}, [
        Node("layernorm", ("x", "w", "b"), "y")], ("y",))
    values = {"x": np.full((2, width), 16777215, np.float32), "w": np.ones(width, np.float32), "b": np.zeros(width, np.float32)}
    np.testing.assert_array_equal(execute(graph, values)["y"], 0)


@pytest.mark.parametrize("eps", [10**400, 1e300, 1e-300, float(np.finfo(np.float32).tiny) / 2,
                                 float(np.nextafter(np.float32(0), np.float32(1))),
                                 0, -1, True, float("nan"), float("inf")])
def test_layernorm_rejects_epsilon_not_positive_finite_float32(eps):
    graph = Graph({"x": TensorSpec((2, 3)), "w": TensorSpec((3,)), "b": TensorSpec((3,))}, [
        Node("layernorm", ("x", "w", "b"), "y", {"eps": eps})], ("y",))
    with pytest.raises(GraphError, match="float32"):
        graph.validate()


def test_benchmark_finite_report_remains_json_serializable():
    values = {"y": np.array([0, 1, 2], np.float32)}
    json.dumps(compare_arrays(values, values, "float32"), allow_nan=False)


@pytest.mark.parametrize("width", [513, 2049])
def test_reference_offset_layernorm_matches_float64_oracle(width):
    graph = Graph({"x": TensorSpec((1, width)), "w": TensorSpec((width,)), "b": TensorSpec((width,))}, [
        Node("layernorm", ("x", "w", "b"), "y")], ("y",))
    x = (16777215 + np.arange(width) % 7 - 3).astype(np.float32).reshape(1, width)
    reference = x.astype(np.float64)
    centered = reference - reference.mean(axis=-1, keepdims=True)
    expected = centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-5)
    actual = execute(graph, {"x": x, "w": np.ones(width, np.float32), "b": np.zeros(width, np.float32)})["y"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("field,value", [("op", []), ("inputs", None), ("attrs", []), ("output", [])])
def test_validation_rechecks_mutated_builder_fields(field, value):
    node = Node("relu", ("x",), "y")
    graph = Graph({"x": TensorSpec((2,))}, [node], ("y",))
    setattr(node, field, value)
    with pytest.raises(GraphError):
        graph.validate()


@pytest.mark.parametrize("field,value", [("inputs", None), ("constants", []), ("nodes", None), ("outputs", [["x"]])])
def test_validation_rechecks_mutated_graph_fields(field, value):
    graph = Graph({"x": TensorSpec((2,))}, [], ("x",))
    setattr(graph, field, value)
    with pytest.raises(GraphError):
        graph.validate()


@pytest.mark.parametrize("inputs", [None, ["x"], {"x": np.ones(2, np.float32), 1: np.ones(2, np.float32)}])
def test_reference_input_signature_errors_keep_layer_error(inputs):
    graph = Graph({"x": TensorSpec((2,))}, [], ("x",))
    with pytest.raises(ExecutionError):
        execute(graph, inputs)


def test_cpu_binding_guards_external_output_shape_without_dispatch():
    graph = Graph({"x": TensorSpec((2, 3))}, [], ("x",))
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    bound = Program(graph).bind({"x": x})
    x.shape = (6,)
    with pytest.raises(ExecutionError, match="metadata|storage"):
        bound.run()


def test_borrowed_constants_do_not_mutate_other_bindings_or_program():
    graph = Graph({}, [], ("c",), {"c": np.array([2, 4], np.float32)})
    program = Program(graph)
    first, second = program.bind({}), program.bind({})
    first.run(copy_outputs=False)["c"][0] = 99
    np.testing.assert_array_equal(second.run()["c"], [2, 4])
    np.testing.assert_array_equal(program.plan.graph.constants["c"], [2, 4])


@pytest.mark.parametrize("fail", [False, True])
def test_profiling_precision_policy_restores_state_even_on_failure(fail):
    from mlgpu.precision import ieee_float32
    state = {"precision": "medium"}
    torch = SimpleNamespace(backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True)),
        get_float32_matmul_precision=lambda: state["precision"],
        set_float32_matmul_precision=lambda value: state.update(precision=value))
    try:
        with ieee_float32(torch):
            assert state["precision"] == "highest"
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
            if fail:
                raise RuntimeError("synthetic profile failure")
    except RuntimeError:
        assert fail
    assert state["precision"] == "medium"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
