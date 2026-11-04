"""Case coverage and settings validation run without GPU imports."""
import pytest

from mlgpu.microbench import CaptureSettings, measure_case, select_cases, tile_configurations


def test_capture_defaults_and_serializable_settings():
    settings = CaptureSettings()
    assert settings.repeats == 64
    assert settings.warmup_replays == 5
    assert settings.samples == 15
    assert settings.to_dict()["setup_warmup"] >= 1


@pytest.mark.parametrize("field,value", [("repeats", 0), ("repeats", True), ("samples", -2),
                                         ("samples", 1.5), ("warmup_replays", 0),
                                         ("setup_warmup", "3")])
def test_invalid_capture_settings_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        CaptureSettings(**{field: value})


@pytest.mark.parametrize("quick", [True, False])
def test_selected_regions_cover_thirty_kernel_cases(quick):
    cases = select_cases(quick)
    assert len(cases) == 30
    assert {case["workload"] for case in cases} == {"matmul", "epilogue", "elementwise", "layernorm_kernel", "softmax_kernel"}
    assert {case["shape_class"] for case in cases} == {"small", "medium", "irregular"}
    assert {case["dtype"] for case in cases} == {"float16", "float32"}
    assert all(case["category"] in {"kernel", "fused_kernel_region"} for case in cases)


def test_quick_mode_requires_a_boolean():
    with pytest.raises(ValueError, match="quick"):
        select_cases(1)


def test_tile_sweep_covers_shape_and_warp_changes():
    configs = tile_configurations()
    assert len(configs) == 7
    assert {config["num_warps"] for config in configs} == {4, 8}
    assert {config["block_k"] for config in configs} == {32, 64}
    assert len([case for case in select_cases() if case["workload"] == "matmul"]) * len(configs) == 42


def test_invalid_measurement_requests_fail_before_importing_gpu():
    case = select_cases()[0]
    with pytest.raises(ValueError, match="implementation"):
        measure_case(case, implementation="silently_fallback")
    with pytest.raises(ValueError, match="settings"):
        measure_case(case, settings={})
    with pytest.raises(ValueError, match="tile overrides"):
        measure_case(case, implementation="pytorch_eager", tile_config={})


@pytest.mark.gpu
def test_small_cuda_graph_capture_validates_replayed_outputs():
    pytest.importorskip("triton")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    case = next(case for case in select_cases() if case["workload"] == "elementwise" and case["shape_class"] == "small" and case["dtype"] == "float32")
    settings = CaptureSettings(repeats=8, warmup_replays=2, samples=3)
    for implementation in ("custom_optimized", "pytorch_eager"):
        result = measure_case(case, implementation, settings)
        assert result["status"] == "ok"
        assert result["correctness"]["passed"]
        assert result["hot_cuda_graph_region"]["count"] == 3
        assert result["hot_cuda_graph_region"]["median_ms"] > 0
        assert len(result["raw_replay_samples_ms"]) == 3
