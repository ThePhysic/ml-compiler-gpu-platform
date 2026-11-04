"""Real PyTorch regressions for coupled precision state; a fake cannot model it."""
from contextlib import nullcontext

import pytest

from mlgpu.errors import BackendUnavailable
from mlgpu.precision import ieee_float32

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def torch_backend():
    from mlgpu.gpu import ensure_available
    try:
        ensure_available()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    import torch
    return torch


def restore(torch, state):
    precision, matmul, cudnn = state
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
    torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize("mode", ["highest", "high", "medium"])
@pytest.mark.parametrize("fail", [False, True])
def test_real_torch_precision_scope_restores_named_mode(torch_backend, mode, fail):
    torch = torch_backend
    original = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision(mode)
        before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                  torch.backends.cudnn.allow_tf32)
        with pytest.raises(RuntimeError, match="synthetic exit") if fail else nullcontext():
            with ieee_float32(torch):
                assert torch.get_float32_matmul_precision() == "highest"
                assert torch.backends.cuda.matmul.allow_tf32 is False
                if fail:
                    raise RuntimeError("synthetic exit")
        after = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32)
        assert after == before
    finally:
        restore(torch, original)


@pytest.mark.parametrize("mode", ["highest", "high", "medium"])
@pytest.mark.parametrize("module_name,selection", [("benchmark", "benchmark_cases"), ("microbench", "select_cases")])
def test_benchmark_entrypoints_restore_real_torch_state(torch_backend, monkeypatch, tmp_path, mode, module_name, selection):
    from importlib import import_module
    module = import_module(f"mlgpu.{module_name}")
    # Empty case selection isolates each entrypoint's public state policy from
    # GPU performance, while still using actual coupled PyTorch backend settings.
    monkeypatch.setattr(module, selection, lambda *args, **kwargs: [])
    torch = torch_backend
    original = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision(mode)
        before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                  torch.backends.cudnn.allow_tf32)
        module.run_suite(tmp_path / f"{module_name}.json", quick=True)
        after = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32)
        assert after == before
    finally:
        restore(torch, original)
