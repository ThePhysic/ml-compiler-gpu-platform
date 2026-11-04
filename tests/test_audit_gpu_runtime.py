"""Independent post-delivery GPU/runtime audit regressions."""
import numpy as np
import pytest

from mlgpu.errors import BackendUnavailable, ExecutionError, ResourceError
from mlgpu.gpu import prepare_launch
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.runtime import Program

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def cuda():
    try:
        from mlgpu.gpu import ensure_available
        ensure_available()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    import torch
    return torch


def test_negative_bit_input_is_rejected(cuda):
    graph = Graph({"x": TensorSpec((3,))}, [Node("relu", ("x",), "y")], ("y",))
    x = cuda.tensor([-1., 2., -3.], device="cuda")
    negative = cuda._neg_view(x)
    assert negative.is_contiguous() and negative.is_neg()
    with pytest.raises(ExecutionError, match="negative/conjugate"):
        Program(graph, backend="gpu").run({"x": negative})


def test_prepared_dispatch_detects_negative_bit_change(cuda):
    x = cuda.tensor([-1., 2., -3.], device="cuda")
    out = cuda.empty_like(x)
    prepared = prepare_launch(Node("relu", ("x",), "y"), [x], out)
    cuda._C._set_neg(x, True)
    with pytest.raises(ExecutionError, match="metadata or storage changed"):
        prepared()


def test_negative_bit_output_is_rejected(cuda):
    x = cuda.tensor([-1., 2., -3.], device="cuda")
    out = cuda._neg_view(cuda.empty_like(x))
    with pytest.raises(ExecutionError, match="negative/conjugate"):
        prepare_launch(Node("relu", ("x",), "y"), [x], out)()


@pytest.mark.parametrize("base", [1e6, 1e8, 16777215.])
@pytest.mark.parametrize("width", [3, 5, 37, 513, 2049])
def test_constant_large_layernorm_returns_affine_bias(cuda, base, width):
    graph = Graph({"x": TensorSpec((1, width)), "w": TensorSpec((width,)), "b": TensorSpec((width,))},
                  [Node("layernorm", ("x", "w", "b"), "y")], ("y",))
    x = cuda.full((1, width), base, device="cuda")
    w = cuda.ones(width, device="cuda")
    b = cuda.zeros(width, device="cuda")
    result = Program(graph, backend="gpu").run({"x": x, "w": w, "b": b})["y"]
    cuda.testing.assert_close(result, cuda.zeros_like(result), atol=1e-5, rtol=0)


def test_constant_layernorm_independent_framework_reference(cuda):
    from mlgpu.reference import execute
    width = 513
    base = 16777215.
    graph = Graph({"x": TensorSpec((1, width)), "w": TensorSpec((width,)), "b": TensorSpec((width,))},
                  [Node("layernorm", ("x", "w", "b"), "y")], ("y",))
    x = cuda.full((1, width), base, device="cuda")
    expected = cuda.nn.functional.layer_norm(x, (width,))
    # The exact normalized constant row is zero. This also checks that the
    # independent framework agrees, rather than accepting a shared reference bug.
    cuda.testing.assert_close(expected, cuda.zeros_like(expected), atol=1e-5, rtol=0)
    cpu = execute(graph, {"x": np.full((1, width), base, np.float32),
                          "w": np.ones(width, np.float32), "b": np.zeros(width, np.float32)})["y"]
    np.testing.assert_allclose(cpu, 0, atol=1e-5, rtol=0)


def test_unused_external_output_metadata_change_requires_rebinding(cuda):
    graph = Graph({"x": TensorSpec((2, 3))}, [], ("x",))
    x = cuda.ones((2, 3), device="cuda")
    bound = Program(graph, backend="gpu").bind({"x": x})
    x.resize_(6)
    with pytest.raises(ExecutionError):
        bound.run()


@pytest.mark.parametrize("dtype,base", [("float16", 1000.), ("float32", 1e6)])
@pytest.mark.parametrize("width", [37, 513])
def test_large_offset_layernorm_variations_match_float64_oracle(cuda, dtype, base, width):
    graph = Graph({"x": TensorSpec((2, width), dtype), "w": TensorSpec((width,), dtype), "b": TensorSpec((width,), dtype)},
                  [Node("layernorm", ("x", "w", "b"), "y")], ("y",))
    offsets = ((np.arange(width) % 13) - 6) / (2 if dtype == "float16" else 8)
    x = np.ascontiguousarray(np.stack([base + offsets, base - offsets]), dtype=dtype)
    w = np.ones(width, dtype=dtype)
    b = np.zeros(width, dtype=dtype)
    precise = x.astype(np.float64)
    centered = precise - precise.mean(-1, keepdims=True)
    expected = (centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)).astype(dtype)
    inputs = {name: cuda.from_numpy(a).cuda() for name, a in {"x": x, "w": w, "b": b}.items()}
    actual = Program(graph, backend="gpu").run(inputs)["y"].cpu().numpy()
    atol = 1e-3 if dtype == "float16" else 3e-5
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.parametrize("eps", [float(np.finfo(np.float32).tiny) / 2,
                                 float(np.nextafter(np.float32(0), np.float32(1))),
                                 float(np.finfo(np.float32).tiny)])
def test_layernorm_epsilon_near_float32_underflow(cuda, eps):
    from mlgpu.errors import GraphError
    graph = Graph({"x": TensorSpec((1, 5)), "w": TensorSpec((5,)), "b": TensorSpec((5,))},
                  [Node("layernorm", ("x", "w", "b"), "y", {"eps": eps})], ("y",))
    if eps < np.finfo(np.float32).tiny:
        with pytest.raises(GraphError):
            Program(graph, backend="gpu")
        return
    program = Program(graph, backend="gpu")
    inputs = {"x": cuda.ones((1, 5), device="cuda"), "w": cuda.ones(5, device="cuda"), "b": cuda.zeros(5, device="cuda")}
    result = program.run(inputs)["y"]
    cuda.testing.assert_close(result, cuda.zeros_like(result), atol=1e-5, rtol=0)


@pytest.mark.parametrize("source", ["unused_input", "constant"])
def test_bound_unused_and_constant_source_metadata_is_guarded(cuda, source):
    constants = {"c": np.ones((2, 3), np.float32)} if source == "constant" else {}
    specs = {"x": TensorSpec((2, 3))}
    if source == "unused_input":
        specs["unused"] = TensorSpec((2, 3))
    graph = Graph(specs, [Node("relu", ("x",), "y")], ("y", "c") if constants else ("y",), constants)
    inputs = {name: cuda.ones(spec.shape, device="cuda") for name, spec in specs.items()}
    bound = Program(graph, backend="gpu").bind(inputs)
    bound.values["c" if constants else "unused"].resize_(6)
    with pytest.raises(ExecutionError, match="metadata or storage changed"):
        bound.run()


def test_matmul_cuda_grid_y_limit_is_supported_or_reported(cuda):
    # A valid ~4 MiB matrix reaches CUDA's y-grid limit with an otherwise legal
    # 16-column tile. Reject a resource limit explicitly rather than leaking a
    # low-level CUDA invalid-configuration exception.
    n = 65535 * 16 + 1
    a = cuda.ones((1, 1), device="cuda")
    b = cuda.ones((1, n), device="cuda")
    out = cuda.empty_like(b)
    try:
        prepared = prepare_launch(Node("matmul", ("a", "b"), "y"), [a, b], out,
                                  {"block_m": 16, "block_n": 16, "block_k": 16, "num_warps": 4})
        prepared()
    except ResourceError:
        return
    cuda.testing.assert_close(out, b)
