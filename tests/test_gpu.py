"""Kernel numerical validation; CPU installations still collect cleanly."""

import os
import subprocess
import sys

import numpy as np
import pytest

from mlgpu.errors import ExecutionError, ResourceError
from mlgpu.ir import Graph, Node, TensorSpec
from mlgpu.reference import evaluate_node


@pytest.fixture(scope="module")
def cuda():
    torch = pytest.importorskip("torch", reason="optional GPU dependency torch is not installed")
    pytest.importorskip("triton", reason="optional GPU dependency Triton is not installed")
    if not torch.cuda.is_available():
        pytest.skip("a CUDA GPU is required")
    from mlgpu.gpu import ensure_available
    ensure_available()
    return torch


def test_backend_import_is_lazy():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    code = "import sys; import mlgpu.gpu; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def _execute(torch, node, arrays, *, config=None, atol=None, rtol=None):
    from mlgpu.gpu import launch
    dtype = arrays[0].dtype.name
    graph = Graph({name: TensorSpec(a.shape, dtype) for name, a in zip(node.inputs, arrays)}, [node], (node.output,))
    spec = graph.specs()[node.output]
    tensors = [torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in arrays]
    output = torch.empty(spec.shape, dtype=getattr(torch, dtype), device="cuda")
    expected = evaluate_node(node, dict(zip(node.inputs, arrays)))
    assert launch(node, tensors, output, config) is None
    actual = output.cpu().numpy()
    tol = (4e-3, 4e-3) if dtype == "float16" else (3e-5, 3e-4)
    np.testing.assert_allclose(actual, expected, atol=tol[0] if atol is None else atol,
                               rtol=tol[1] if rtol is None else rtol,
                               err_msg=f"kernel {node.op} / {dtype} / {spec.shape}")
    return actual, expected


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("op", ["add", "mul", "relu", "gelu", "identity"])
def test_elementwise_tails_and_broadcast(cuda, dtype, op):
    rng = np.random.default_rng(191)
    x = rng.normal(size=(3, 5, 67)).astype(dtype)
    arrays = [x]
    names = ("x",)
    if op in ("add", "mul"):
        arrays.append(rng.normal(size=(1, 5, 1)).astype(dtype))
        names += ("b",)
    _execute(cuda, Node(op, names, "y"), arrays)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_relu_propagates_nan_and_infinities(cuda, dtype):
    x = np.array([[np.nan, np.inf, -np.inf, -0.0, -1.0, 1.0]], dtype=dtype)
    actual, expected = _execute(cuda, Node("relu", ("x",), "y"), [x])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width", [1, 37, 257, 2049])
def test_softmax_stability_and_irregular_width(cuda, dtype, width):
    rng = np.random.default_rng(20)
    x = (rng.normal(size=(3, width)) * 19 + 1000).astype(dtype)
    actual, _ = _execute(cuda, Node("softmax", ("x",), "y"), [x])
    np.testing.assert_allclose(actual.astype(np.float32).sum(-1), 1, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width", [1, 37, 513, 2049])
def test_layernorm_rows_and_constant_input(cuda, dtype, width):
    rng = np.random.default_rng(30)
    x = rng.normal(size=(3, width)).astype(dtype)
    x[0] = 100
    weight = rng.uniform(0.5, 1.5, size=width).astype(dtype)
    bias = rng.normal(size=width).astype(dtype)
    actual, _ = _execute(cuda, Node("layernorm", ("x", "w", "b"), "y", {"eps": 1e-4}), [x, weight, bias])
    np.testing.assert_array_equal(actual[0], bias)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 37, 29), (64, 64, 64), (65, 97, 33)])
@pytest.mark.parametrize("activation", ["none", "relu", "gelu"])
def test_matmul_epilogue_all_dimension_tails(cuda, dtype, shape, activation):
    m, k, n = shape
    rng = np.random.default_rng(55)
    a = (rng.normal(size=(m, k)) * 0.3).astype(dtype)
    b = (rng.normal(size=(k, n)) * 0.3).astype(dtype)
    bias = rng.normal(size=(1, n)).astype(dtype)
    node = Node("fused_matmul", ("a", "b", "bias"), "y", {"activation": activation})
    _execute(cuda, node, [a, b, bias])


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("tiles", [(16, 32, 16, 4), (32, 64, 32, 4), (64, 32, 16, 8)])
def test_matmul_tile_variants(cuda, dtype, tiles):
    rng = np.random.default_rng(6)
    arrays = [rng.normal(size=(31, 71)).astype(dtype), rng.normal(size=(71, 93)).astype(dtype)]
    config = dict(zip(("block_m", "block_n", "block_k", "num_warps"), tiles))
    _execute(cuda, Node("matmul", ("a", "b"), "y"), arrays, config=config)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_fused_ssa_external_broadcast_and_dtype_barriers(cuda, dtype):
    rng = np.random.default_rng(12)
    x = rng.uniform(-2, 2, size=(3, 67)).astype(dtype)
    b = rng.uniform(-1, 1, size=(67,)).astype(dtype)
    s = rng.uniform(0.1, 0.9, size=(3, 1)).astype(dtype)
    node = Node("fused_elementwise", ("x", "b", "s"), "y", {"steps": [
        {"op": "add", "inputs": ["x", "b"], "output": "u"},
        {"op": "mul", "inputs": ["u", "s"], "output": "v"},
        {"op": "add", "inputs": ["v", "u"], "output": "t"},
        {"op": "relu", "inputs": ["t"], "output": "y"},
    ]})
    actual, expected = _execute(cuda, node, [x, b, s])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_maximal_fused_chain_accepts_seventeen_external_inputs(cuda, dtype):
    rng = np.random.default_rng(118)
    names = tuple(f"x{i}" for i in range(17))
    arrays = [rng.uniform(-0.2, 0.2, size=(3, 67)).astype(dtype) for _ in names]
    steps = []
    previous = names[0]
    for index in range(16):
        result = "y" if index == 15 else f"t{index}"
        steps.append({"op": "add", "inputs": [previous, names[index + 1]], "output": result})
        previous = result
    node = Node("fused_elementwise", names, "y", {"steps": steps})
    actual, expected = _execute(cuda, node, arrays)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
def test_half_epilogue_preserves_matmul_and_bias_rounding(cuda):
    from mlgpu.gpu import launch
    rng = np.random.default_rng(103)
    a = cuda.from_numpy(rng.normal(size=(17, 19)).astype("float16")).cuda()
    b = cuda.from_numpy(rng.normal(size=(19, 31)).astype("float16")).cuda()
    bias = cuda.from_numpy(rng.normal(size=(31,)).astype("float16")).cuda()
    product = cuda.empty((17, 31), dtype=cuda.float16, device="cuda")
    separate = cuda.empty_like(product)
    fused = cuda.empty_like(product)
    launch(Node("matmul", ("a", "b"), "p"), [a, b], product)
    launch(Node("add", ("p", "bias"), "y"), [product, bias], separate)
    launch(Node("fused_matmul", ("a", "b", "bias"), "y"), [a, b, bias], fused)
    np.testing.assert_array_equal(fused.cpu().numpy(), separate.cpu().numpy())


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("keep", [False, True])
def test_reduction_nonfinal_axes(cuda, dtype, axis, keep):
    x = np.random.default_rng(22).normal(size=(3, 7, 19)).astype(dtype)
    _execute(cuda, Node("sum", ("x",), "y", {"axis": axis, "keepdims": keep}), [x])


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_materialized_layout_operations(cuda, dtype):
    x = np.random.default_rng(2).normal(size=(3, 5, 7)).astype(dtype)
    for node in (Node("transpose", ("x",), "y", {"axes": [1, 2, 0]}),
                 Node("transpose", ("x",), "y"),
                 Node("reshape", ("x",), "y", {"shape": [15, 7]})):
        actual, expected = _execute(cuda, node, [x])
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
def test_backend_rejects_aliases_layouts_shapes_and_resource_limits(cuda):
    from mlgpu.gpu import launch, MAX_REDUCTION_WIDTH
    x = cuda.zeros((3, 5), device="cuda")
    node = Node("relu", ("x",), "y")
    with pytest.raises(ExecutionError, match="overlap"):
        launch(node, [x], x)
    with pytest.raises(ExecutionError, match="contiguous"):
        launch(node, [x.T], cuda.empty((5, 3), device="cuda"))
    with pytest.raises(ExecutionError, match="output shape"):
        launch(node, [x], cuda.empty((15,), device="cuda"))
    with pytest.raises(ExecutionError, match="CUDA tensors"):
        launch(node, [x.cpu()], cuda.empty_like(x))
    with pytest.raises(ResourceError, match="block_m"):
        launch(Node("matmul", ("a", "b"), "y"), [x, x.T.contiguous()],
               cuda.empty((3, 3), device="cuda"), {"block_m": 3})
    wide = cuda.zeros((1, MAX_REDUCTION_WIDTH + 1), device="cuda")
    with pytest.raises(ResourceError, match="reduction width"):
        launch(Node("softmax", ("x",), "y"), [wide], cuda.empty_like(wide))


@pytest.mark.gpu
def test_prepared_launch_freezes_ir_and_configuration(cuda):
    from mlgpu.gpu import prepare_launch
    rng = np.random.default_rng(81)
    arrays = [rng.normal(size=(7, 11)).astype("float32"), rng.normal(size=(11, 13)).astype("float32")]
    inputs = [cuda.from_numpy(a).cuda() for a in arrays]
    output = cuda.empty((7, 13), device="cuda")
    node = Node("fused_matmul", ("a", "b"), "y", {"activation": "relu"})
    config = {"block_m": 16, "block_n": 32, "block_k": 16, "num_warps": 4}
    prepared = prepare_launch(node, inputs, output, config)
    node.op = "unsupported"
    node.attrs["activation"] = "gelu"
    config["block_m"] = 3
    assert prepared() is None
    metadata = prepared.kernel_metadata
    assert metadata["kernel"] == "matmul_epilogue"
    assert metadata["registers_per_thread"] > 0
    assert metadata["shared_memory_bytes"] >= 0
    assert metadata["num_warps"] == 4
    assert metadata["grid"] == [1, 1]
    expected = np.maximum(arrays[0] @ arrays[1], 0)
    np.testing.assert_allclose(output.cpu().numpy(), expected, atol=1e-5, rtol=1e-5)
    with pytest.raises(ResourceError, match="block_m"):
        prepare_launch(Node("matmul", ("a", "b"), "y"), inputs, output, config)


@pytest.mark.gpu
def test_prepared_launch_rejects_stale_storage_but_accepts_new_data(cuda):
    from mlgpu.gpu import prepare_launch
    x = cuda.ones((3, 5), device="cuda")
    y = cuda.empty_like(x)
    prepared = prepare_launch(Node("relu", ("x",), "y"), [x], y)
    x.fill_(2)
    prepared()
    np.testing.assert_array_equal(y.cpu().numpy(), np.full((3, 5), 2, dtype=np.float32))
    x.resize_(15)
    with pytest.raises(ExecutionError, match="metadata or storage changed"):
        prepared()
