"""Hardware-required integration tests; CPU skips cannot satisfy GPU acceptance."""
import numpy as np
import pytest

from mlgpu.errors import BackendUnavailable, ExecutionError
from mlgpu.examples import make_inputs, workload
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.reference import execute
from mlgpu.runtime import Program

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def torch_gpu():
    try:
        from mlgpu.gpu import ensure_available
        ensure_available()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch


def compare(torch, graph):
    inputs = make_inputs(graph, 14)
    expected = execute(graph, inputs)
    tensors = {name: torch.from_numpy(value).cuda() for name, value in inputs.items()}
    for optimize in (False, True):
        program = Program(graph, backend="gpu", optimize=optimize)
        bound = program.bind(tensors)
        actual = bound.run()
        # Run again to expose memory reuse and output lifetime mistakes.
        bound.run(copy_outputs=False)
        torch.cuda.synchronize()
        for name in graph.outputs:
            result = actual[name].cpu().numpy()
            atol, rtol = (.02, .02) if result.dtype == np.float16 else (3e-4, 3e-4)
            np.testing.assert_allclose(result, expected[name], atol=atol, rtol=rtol)
        for name, allocation in program.plan.memory.allocations.items():
            assert bound.values[name].data_ptr() - bound.arena.data_ptr() == allocation.offset


@pytest.mark.parametrize("name", ["epilogue", "norm", "mlp", "softmax", "elementwise", "matmul"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_compiled_graph_gpu_path(torch_gpu, name, dtype):
    compare(torch_gpu, workload(name, 13, 37, 53, dtype))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_gpu_layout_reductions_and_pinned_branch_outputs(torch_gpu, dtype):
    graph = Graph({"x": TensorSpec((2, 3, 5), dtype), "b": TensorSpec((5,), dtype)}, [
        Node("add", ("x", "b"), "a"), Node("relu", ("a",), "r"),
        Node("transpose", ("r",), "t", {"axes": [2, 0, 1]}),
        Node("sum", ("t",), "s", {"axis": 1, "keepdims": False}),
        Node("reshape", ("s",), "flat", {"shape": [15]}),
        Node("mul", ("r", "a"), "product")], ("a", "flat", "product", "x"))
    compare(torch_gpu, graph)


def test_gpu_constant_folding_and_external_outputs(torch_gpu):
    c = np.array([1, -2, 3], np.float32)
    graph = Graph({"x": TensorSpec((3,))}, [Node("add", ("c", "c"), "folded"), Node("add", ("folded", "x"), "y")], ("x", "c", "folded", "y"), {"c": c})
    compare(torch_gpu, graph)


def test_gpu_rejects_invalid_inputs(torch_gpu):
    graph = workload("epilogue", 3, 5, 7)
    p = Program(graph, backend="gpu")
    with pytest.raises(ExecutionError):
        p.bind(make_inputs(graph))
    with pytest.raises(ExecutionError):
        p.bind({})


def test_gpu_binding_rejects_cross_stream_arena_reuse(torch_gpu):
    graph = workload("elementwise", 3, 5, 7)
    inputs = {name: torch_gpu.from_numpy(value).cuda() for name, value in make_inputs(graph).items()}
    bound = Program(graph, backend="gpu").bind(inputs)
    bound.run(copy_outputs=False)
    other = torch_gpu.cuda.Stream()
    with torch_gpu.cuda.stream(other):
        with pytest.raises(ExecutionError, match="binding CUDA stream"):
            bound.run()
    bound.run()
    torch_gpu.cuda.synchronize()


def test_framework_reference_matches_numpy(torch_gpu):
    from mlgpu.torch_reference import make_callable
    for name in ("epilogue", "norm", "mlp", "softmax", "elementwise"):
        graph = workload(name, 7, 19, 31, "float16")
        inputs = make_inputs(graph, 9)
        expected = execute(graph, inputs)
        tensors = {name: torch_gpu.from_numpy(value).cuda() for name, value in inputs.items()}
        fn, args = make_callable(graph, tensors)
        actual = fn(*args)
        torch_gpu.cuda.synchronize()
        for name, value in zip(graph.outputs, actual):
            np.testing.assert_allclose(value.cpu().numpy(), expected[name], atol=.02, rtol=.02)


def test_independent_kernel_rejects_ambiguous_repeated_names(torch_gpu):
    from mlgpu.gpu import prepare_launch
    a = torch_gpu.ones((2, 3), device="cuda")
    b = torch_gpu.zeros_like(a)
    out = torch_gpu.empty_like(a)
    with pytest.raises(ExecutionError, match="Repeated GPU input names"):
        prepare_launch(Node("add", ("x", "x"), "y"), [a, b], out)
    prepare_launch(Node("add", ("x", "x"), "y"), [a, a], out)()
    torch_gpu.cuda.synchronize()
    torch_gpu.testing.assert_close(out, a + a)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_preexisting_fused_region_alias_scope_on_gpu(torch_gpu, dtype):
    graph = Graph({"x": TensorSpec((3, 7), dtype)}, [
        Node("identity", ("x",), "a"),
        Node("fused_elementwise", ("a",), "y", {"steps": [
            {"op": "relu", "inputs": ["a"], "output": "x"},
            {"op": "add", "inputs": ["x", "a"], "output": "y"}]})], ("y",))
    compare(torch_gpu, graph)
