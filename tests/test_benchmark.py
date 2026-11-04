"""CPU tests verify reporting math, case coverage and numerical failure handling."""
import json
import sys

import numpy as np
import pytest

from mlgpu.benchmark import benchmark_cases, compare_arrays, operation_metrics, summarize_samples, tolerances, _write_report
from mlgpu.errors import ExecutionError
from mlgpu.examples import workload


def test_statistics_preserve_samples_and_variability():
    result = summarize_samples([1, 2, 3, 4, 5])
    assert result["samples_ms"] == [1, 2, 3, 4, 5]
    assert result["count"] == 5
    assert result["median_ms"] == 3
    assert result["mean_ms"] == 3
    assert result["p05_ms"] == pytest.approx(1.2)
    assert result["p95_ms"] == pytest.approx(4.8)
    assert result["stdev_ms"] == pytest.approx(np.sqrt(2.5))
    assert summarize_samples([0])["coefficient_of_variation"] == 0


@pytest.mark.parametrize("samples", [[], [-1], [float("nan")], [float("inf")], [[1, 2]]])
def test_invalid_sample_sequences_rejected(samples):
    with pytest.raises(ValueError, match="samples"):
        summarize_samples(samples)


def test_algorithmic_metrics_units_and_explicit_scope():
    graph = workload("matmul", m=2, k=3, n=5, dtype="float32")
    result = operation_metrics(graph, .1)
    assert result["matmul_flops"] == 60
    assert result["minimum_external_bytes"] == (6 + 15 + 10) * 4
    assert result["logical_plan_bytes"] == result["minimum_external_bytes"]
    assert result["effective_matmul_gflop_s"] == pytest.approx(60 / 100000)
    assert "not measured DRAM" in result["interpretation"]


@pytest.mark.parametrize("latency", [0, -1, float("nan"), float("inf")])
def test_invalid_metric_latencies_rejected(latency):
    with pytest.raises(ValueError, match="latency"):
        operation_metrics(workload(), latency)


def test_matmul_free_workload_does_not_claim_matmul_throughput():
    assert operation_metrics(workload("norm"), .1)["effective_matmul_gflop_s"] is None


@pytest.mark.parametrize("quick", [False, True])
def test_case_matrix_covers_shapes_precisions_and_independent_kernels(quick):
    cases = benchmark_cases(quick)
    assert len(cases) == 48
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["dtype"] for case in cases} == {"float16", "float32"}
    assert {case["shape_class"] for case in cases} == {"small", "medium", "irregular"}
    assert {case["category"] for case in cases} == {"kernel", "fused_kernel_region", "graph"}
    for case in cases:
        case["graph"].validate()
        if case["category"] == "kernel":
            assert len(case["graph"].nodes) == 1


def test_correctness_reports_errors_and_never_accepts_nan():
    target = {"y": np.array([1., -2.], dtype=np.float32)}
    assert compare_arrays(target, target, "float32")["passed"]
    with pytest.raises(ExecutionError, match="numerical mismatch"):
        compare_arrays({"y": np.array([1., float("nan")], dtype=np.float32)}, target, "float32")
    with pytest.raises(ExecutionError, match="numerical mismatch"):
        compare_arrays({"y": np.array([1., -1.], dtype=np.float32)}, target, "float32")
    with pytest.raises(ExecutionError, match="shape/dtype"):
        compare_arrays({"y": target["y"].astype(np.float16)}, target, "float32")
    with pytest.raises(ExecutionError, match="output names"):
        compare_arrays({"z": target["y"]}, target, "float32")


def test_precision_tolerances_are_explicit():
    assert tolerances("float16")["atol"] > tolerances("float32")["atol"]
    with pytest.raises(ValueError):
        tolerances("bfloat16")


def test_report_is_standard_json_and_rejects_nonfinite_numbers(tmp_path):
    path = tmp_path / "nested" / "report.json"
    _write_report(path, {"status": "ok", "stats": summarize_samples([.1, .2])})
    assert json.loads(path.read_text())["status"] == "ok"
    with pytest.raises(ValueError):
        _write_report(path, {"bad": float("nan")})


def test_import_does_not_require_gpu_dependencies():
    # Module-level helpers run successfully in the CPU-only test environment.
    assert "mlgpu.benchmark" in sys.modules
    assert summarize_samples([1])["count"] == 1
