"""Hot-cache CUDA Graph measurements that remove Python launch gaps.

These measurements complement, rather than replace, the allocation-inclusive
runtime and ordinary event measurements in :mod:`mlgpu.benchmark`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import time

from .benchmark import (benchmark_cases, compare_arrays, operation_metrics,
                        summarize_samples, _error, _metadata, _nvidia_snapshot,
                        _numpy_outputs, _write_report)
from .examples import make_inputs
from .reference import execute


@dataclass(frozen=True)
class CaptureSettings:
    repeats: int = 64
    warmup_replays: int = 5
    samples: int = 15
    setup_warmup: int = 3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self):
        return asdict(self)


def select_cases(quick=True):
    """Thirty primitive/fused regions: five kinds × three shapes × two dtypes."""
    if type(quick) is not bool:
        raise ValueError("quick must be a bool")
    return [case for case in benchmark_cases(quick=quick)
            if case["category"] in {"kernel", "fused_kernel_region"}]


def tile_configurations():
    return [{"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": warps}
            for bm, bn, bk, warps in [(16, 32, 32, 4), (32, 64, 32, 4),
                                     (64, 64, 32, 4), (32, 128, 32, 4),
                                     (64, 128, 32, 8), (32, 64, 64, 4),
                                     (64, 64, 32, 8)]]


def measure_case(case, implementation="custom_optimized", settings=None, tile_config=None):
    """Capture and validate one region, preserving its fixed storage until done.

    The custom binding is created on its capture stream. All graph replays and
    timing events use that same stream. No GPU work is launched at import time.
    """
    if implementation not in {"custom_optimized", "pytorch_eager"}:
        raise ValueError("implementation must be custom_optimized or pytorch_eager")
    settings = CaptureSettings() if settings is None else settings
    if not isinstance(settings, CaptureSettings):
        raise ValueError("settings must be CaptureSettings")
    if tile_config is not None and (implementation != "custom_optimized" or case["workload"] != "matmul"):
        raise ValueError("tile overrides require a custom matmul case")
    from .gpu import ensure_available
    ensure_available()
    import torch
    from .runtime import Program
    from .torch_reference import make_callable

    setup_start = time.perf_counter()
    graph = case["graph"]
    arrays = make_inputs(graph, seed=123)
    expected = execute(graph, arrays)
    inputs = {name: torch.from_numpy(array).to("cuda:0") for name, array in arrays.items()}
    upstream = torch.cuda.current_stream(0)
    stream = torch.cuda.Stream(device=0)
    stream.wait_stream(upstream)
    binding = None
    program = None
    compile_ms = 0.0
    with torch.inference_mode(), torch.cuda.stream(stream):
        if implementation == "custom_optimized":
            program = Program(graph, backend="gpu", optimize=True)
            compile_ms = program.compile_ms
            if tile_config is not None:
                program.plan.tiles["y"] = dict(tile_config)
            # The runtime rejects dispatch on a different stream from binding.
            binding = program.bind(inputs)
            fn = lambda: binding.run(copy_outputs=False)
        else:
            eager, args = make_callable(graph, inputs)
            fn = lambda: eager(*args)

        first_start = time.perf_counter()
        initial_outputs = fn()
        stream.synchronize()
        first_use_ms = (time.perf_counter() - first_start) * 1000
        del initial_outputs
        for _ in range(settings.setup_warmup - 1):
            fn()
        stream.synchronize()

        cuda_graph = torch.cuda.CUDAGraph()
        capture_start = time.perf_counter()
        last_outputs = None
        with torch.cuda.graph(cuda_graph, stream=stream):
            for _ in range(settings.repeats):
                last_outputs = fn()
        stream.synchronize()
        capture_ms = (time.perf_counter() - capture_start) * 1000
        setup_ms = (time.perf_counter() - setup_start) * 1000

        # cuda_graph, inputs, last_outputs, fn, and the custom binding all remain
        # strongly referenced until every replay, event and correctness check ends.
        for _ in range(settings.warmup_replays):
            cuda_graph.replay()
        stream.synchronize()
        samples = []
        for _ in range(settings.samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            cuda_graph.replay()
            end.record(stream)
            end.synchronize()
            samples.append(start.elapsed_time(end))
        stream.synchronize()
        correctness = compare_arrays(_numpy_outputs(graph, last_outputs), expected, case["dtype"])
        region = summarize_samples(value / settings.repeats for value in samples)
        timed_graph = program.plan.graph if program is not None else graph
        result = {"status": "ok", "implementation": implementation,
                  "settings": settings.to_dict(), "compile_ms": compile_ms,
                  "first_use_wall_ms": first_use_ms, "capture_wall_ms": capture_ms,
                  "setup_wall_ms": setup_ms, "raw_replay_samples_ms": samples,
                  "hot_cuda_graph_region": region, "correctness": correctness,
                  "metrics": operation_metrics(timed_graph, region["median_ms"]),
                  "scope": "hot-cache repeated fixed-input CUDA Graph device-region latency; excludes Python dispatch, allocations, transfers and JIT",
                  "tiles": program.plan.tiles if program is not None else None,
                  "kernel_resources": [dict(dispatch.kernel_metadata) for dispatch in binding._dispatches] if binding is not None else None}
    return result


def run_suite(output_path, quick=True):
    """Measure 30 regions against equally captured eager execution and 42 tiles."""
    cases = select_cases(quick)
    settings = CaptureSettings(samples=15 if quick else 50)
    report = {"schema_version": 1, "status": "running", "metadata": _metadata(),
              "settings": {**settings.to_dict(), "quick": quick, "seed": 123},
              "methodology": {"scope": "hot-cache repeated fixed-input CUDA Graph device-region latency",
                              "capture": "64 independent fixed-input invocations per graph, side-stream binding and capture, five warmup replays",
                              "timing": "CUDA event milliseconds per full graph replay divided by 64; no Python between captured invocations",
                              "comparison": "custom optimized and PyTorch eager captured identically; eager host allocations become fixed graph-pool storage",
                              "limitations": "Inputs and buffers are reused; hot caches and CUDA Graph scheduling differ from production dispatch. Replay overhead is amortized, not assumed zero. Complements full runtime measurements; no hardware bandwidth/utilization counters."},
              "cases": [], "tile_sweep": []}
    _write_report(output_path, report)
    try:
        from .gpu import ensure_available
        ensure_available()
        import torch
    except Exception as exc:
        report.update({"status": "unavailable", "error": _error(exc)})
        _write_report(output_path, report)
        raise
    original_precision = torch.get_float32_matmul_precision()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["metadata"] = _metadata(torch)
    start_suite = time.perf_counter()
    context = {}
    try:
        for index, case in enumerate(cases):
            print(f"microbenchmark {index + 1}/{len(cases)} {case['id']}", flush=True)
            entry = {key: value for key, value in case.items() if key != "graph"}
            entry["implementations"] = {}
            order = ["custom_optimized", "pytorch_eager"] if index % 2 == 0 else ["pytorch_eager", "custom_optimized"]
            entry["measurement_order"] = order
            report["cases"].append(entry)
            for implementation in order:
                gc.collect()
                context = {"case": case["id"], "implementation": implementation}
                entry["implementations"][implementation] = measure_case(case, implementation, settings)
                _write_report(output_path, report)
            eager_ms = entry["implementations"]["pytorch_eager"]["hot_cuda_graph_region"]["median_ms"]
            custom_ms = entry["implementations"]["custom_optimized"]["hot_cuda_graph_region"]["median_ms"]
            entry["custom_speedup_vs_eager"] = eager_ms / custom_ms
            _write_report(output_path, report)
        for case in cases:
            if case["workload"] != "matmul":
                continue
            for config in tile_configurations():
                gc.collect()
                context = {"case": case["id"], "tile_config": config}
                result = measure_case(case, settings=settings, tile_config=config)
                report["tile_sweep"].append({"case": case["id"], "config": config, **result})
                _write_report(output_path, report)
        report["status"] = "ok"
    except Exception as exc:
        report.update({"status": "failed", "failure_context": context, "error": _error(exc)})
        raise
    finally:
        report["elapsed_wall_seconds"] = time.perf_counter() - start_suite
        report["nvidia_smi_after"] = _nvidia_snapshot()
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn
        _write_report(output_path, report)
    return report
