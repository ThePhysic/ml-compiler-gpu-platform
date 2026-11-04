"""Reproducible real-GPU measurements; imports remain safe without CUDA/PyTorch."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

import numpy as np

from .errors import ExecutionError
from .examples import make_inputs, workload
from .ir import Graph, Node


def summarize_samples(samples) -> dict:
    """Statistics retain all samples; standard deviation uses Bessel's correction."""
    values = np.asarray(list(samples), dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("samples must be a nonempty sequence of finite nonnegative milliseconds")
    mean = float(values.mean())
    deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"samples_ms": values.tolist(), "count": len(values), "mean_ms": mean,
            "median_ms": float(np.median(values)), "p05_ms": float(np.quantile(values, .05)),
            "p95_ms": float(np.quantile(values, .95)), "min_ms": float(values.min()),
            "max_ms": float(values.max()), "stdev_ms": deviation,
            "coefficient_of_variation": deviation / mean if mean else 0.0}


def operation_metrics(graph: Graph, latency_ms: float) -> dict:
    """Algorithmic work/traffic proxies, explicitly not hardware-counter metrics."""
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError("latency must be finite and positive")
    specs = graph.specs()
    flops = 0
    logical_bytes = 0
    for node in graph.nodes:
        logical_bytes += sum(specs[x].nbytes for x in node.inputs) + specs[node.output].nbytes
        if node.op in {"matmul", "fused_matmul"}:
            m, k = specs[node.inputs[0]].shape
            n = specs[node.inputs[1]].shape[1]
            flops += 2 * m * n * k
    external_bytes = sum(s.nbytes for name, s in specs.items() if name in graph.inputs or name in graph.constants)
    external_bytes += sum(specs[name].nbytes for name in graph.outputs)
    return {"matmul_flops": flops, "logical_plan_bytes": logical_bytes,
            "minimum_external_bytes": external_bytes,
            "effective_matmul_gflop_s": flops / (latency_ms * 1e6) if flops else None,
            "logical_plan_gb_s": logical_bytes / (latency_ms * 1e6),
            "minimum_external_gb_s": external_bytes / (latency_ms * 1e6),
            "interpretation": "algorithmic work divided by measured latency; not measured DRAM bandwidth, utilization or hardware counters"}


def tolerances(dtype: str) -> dict:
    if dtype == "float16":
        return {"rtol": .01, "atol": .01}
    if dtype == "float32":
        return {"rtol": .001, "atol": .0002}
    raise ValueError(f"unsupported benchmark dtype {dtype}")


def compare_arrays(actual: dict, expected: dict, dtype: str) -> dict:
    if set(actual) != set(expected):
        raise ExecutionError(f"benchmark output names differ: {set(actual)} versus {set(expected)}")
    tolerance = tolerances(dtype)
    details = {}
    for name, target in expected.items():
        output = np.asarray(actual[name])
        if output.shape != target.shape or output.dtype != target.dtype:
            raise ExecutionError(f"benchmark output {name}: shape/dtype mismatch")
        difference = np.abs(output.astype(np.float64) - target.astype(np.float64))
        if not np.allclose(output, target, equal_nan=False, **tolerance):
            raise ExecutionError(f"benchmark numerical mismatch at {name}; max absolute error {float(difference.max())}")
        scaled = difference / (tolerance["atol"] + tolerance["rtol"] * np.abs(target.astype(np.float64)))
        details[name] = {"max_absolute_error": float(difference.max()), "max_tolerance_fraction": float(scaled.max())}
    return {"passed": True, "tolerances": tolerance, "outputs": details}


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _nvidia_snapshot():
    executable = shutil.which("nvidia-smi")
    if not executable and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if not executable:
        return {"status": "unavailable", "reason": "nvidia-smi not found"}
    fields = "name,driver_version,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate"
    try:
        result = subprocess.run([executable, f"--query-gpu={fields}", "--format=csv"], capture_output=True, text=True, timeout=10)
        return {"status": "ok" if result.returncode == 0 else "unavailable", "csv": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _metadata(torch=None):
    versions = {}
    for name in ("numpy", "torch", "triton", "mlgpu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    cpu = platform.processor()
    if not cpu and Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "os": platform.platform(),
              "python": platform.python_version(), "cpu": cpu, "machine": platform.machine(),
              "versions": versions, "nvidia_smi": _nvidia_snapshot()}
    if torch is not None:
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                         "compute_capability": list(torch.cuda.get_device_capability(0)),
                         "multiprocessors": props.multi_processor_count,
                         "torch_cuda_version": torch.version.cuda, "device": "cuda:0"}
        result["precision"] = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                               "float32_matmul_precision": torch.get_float32_matmul_precision()}
    return result


def _event_samples(torch, fn, repetitions, calls_per_sample=1):
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(calls_per_sample):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / calls_per_sample)
    return summarize_samples(samples)


def _wall_samples(torch, fn, repetitions):
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return summarize_samples(samples)


def _numpy_outputs(graph, outputs):
    if not isinstance(outputs, dict):
        outputs = dict(zip(graph.outputs, outputs))
    return {name: a.detach().cpu().numpy().copy() for name, a in outputs.items()}


def _first_call(torch, fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def _measure(torch, fn, graph, expected, framework_expected, dtype, settings):
    actual, first_use = _first_call(torch, fn)
    arrays = _numpy_outputs(graph, actual)
    correctness = {"numpy_reference": compare_arrays(arrays, expected, dtype),
                   "pytorch_eager": compare_arrays(arrays, framework_expected, dtype)}
    del actual, arrays
    for _ in range(settings["warmup"]):
        fn()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    event = _event_samples(torch, fn, settings["repetitions"], settings["calls_per_sample"])
    wall = _wall_samples(torch, fn, settings["host_repetitions"])
    memory = {"resident_allocated_bytes": resident, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "incremental_peak_bytes": torch.cuda.max_memory_allocated() - resident,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "scope": "PyTorch allocator process totals for warmed event and wall samples; excludes driver/JIT/external allocations"}
    return {"status": "ok", "first_use_wall_ms": first_use, "correctness": correctness,
            "warm_cuda": event, "warm_host_synchronized": wall, "memory": memory}


def _benchmark_custom(torch, graph, inputs, expected, framework_expected, dtype, optimize, settings):
    from .runtime import Program
    program = Program(graph, backend="gpu", optimize=optimize)
    start = time.perf_counter()
    binding = program.bind(inputs)
    torch.cuda.synchronize()
    bind_ms = (time.perf_counter() - start) * 1000
    result = _measure(torch, lambda: binding.run(copy_outputs=False), graph, expected, framework_expected, dtype, settings)
    result.update({"compile_ms": program.compile_ms, "bind_wall_ms": bind_ms,
                   "kernel_launches_per_run": len(program.plan.nodes), "arena_bytes": program.plan.memory.peak_bytes,
                   "naive_arena_bytes": program.plan.memory.naive_bytes, "tiles": program.plan.tiles,
                   "kernel_resources": [dict(dispatch.kernel_metadata) for dispatch in binding._dispatches],
                   "metrics": operation_metrics(program.plan.graph, result["warm_cuda"]["median_ms"])})
    result["end_to_end_host_synchronized"] = _wall_samples(torch, lambda: program.run(inputs), settings["host_repetitions"])
    result["end_to_end_scope"] = "validated already-resident GPU inputs, fresh arena/constants, dispatch, output clones and synchronization; excludes H2D/D2H transfers"
    return result


def benchmark_cases(quick=False):
    """Small, medium and irregular static shapes in both supported precisions."""
    shapes = [("small", 16, 32, 64), ("medium", 128, 256, 384), ("irregular", 37, 93, 127)]
    if not quick:
        shapes = [("small", 32, 64, 96), ("medium", 256, 512, 768), ("irregular", 113, 259, 509)]
    cases = []
    for dtype in ("float16", "float32"):
        for label, m, k, n in shapes:
            for name in ("matmul", "epilogue", "elementwise", "norm", "softmax", "mlp"):
                graph = workload(name, m=m, k=k, n=n, dtype=dtype)
                category = "kernel" if name == "matmul" else ("fused_kernel_region" if name in {"epilogue", "elementwise"} else "graph")
                cases.append({"id": f"{name}-{label}-{dtype}", "workload": name, "category": category,
                              "shape_class": label, "dimensions": {"m": m, "k": k, "n": n}, "dtype": dtype, "graph": graph})
            norm = workload("norm", m=m, n=n, dtype=dtype)
            norm.inputs.pop("r")
            norm.nodes = [Node("layernorm", ("x", "weight", "bias"), "y", {"eps": 1e-5})]
            softmax = workload("softmax", m=m, n=n, dtype=dtype)
            softmax.inputs.pop("b")
            softmax.nodes = [Node("softmax", ("x",), "y")]
            for name, graph in (("layernorm_kernel", norm), ("softmax_kernel", softmax)):
                cases.append({"id": f"{name}-{label}-{dtype}", "workload": name, "category": "kernel",
                              "shape_class": label, "dimensions": {"m": m, "n": n}, "dtype": dtype, "graph": graph.validate()})
    return cases


def _error(exc):
    return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=6)}


def _optional_compile_error(exc):
    # These are compilation capabilities of an optional comparator, not a custom
    # runtime fallback. Numerical and execution failures after compilation fail.
    return type(exc).__name__ in {"Unsupported", "BackendCompilerFailed", "CacheLimitExceeded", "RecompileError"}


def _tile_sweep(torch, cases, settings):
    from .reference import execute
    from .runtime import Program
    results = []
    configurations = [(16, 32, 32, 4), (32, 64, 32, 4), (64, 64, 32, 4),
                      (32, 128, 32, 4), (64, 128, 32, 8), (32, 64, 64, 4), (64, 64, 32, 8)]
    for case in cases:
        if case["workload"] != "matmul":
            continue
        graph = case["graph"]
        arrays = make_inputs(graph, seed=123)
        expected = execute(graph, arrays)
        inputs = {name: torch.from_numpy(a).cuda() for name, a in arrays.items()}
        program = Program(graph, backend="gpu")
        for bm, bn, bk, warps in configurations:
            config = {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": warps}
            program.plan.tiles["y"] = config
            entry = {"case": case["id"], "config": config}
            try:
                # Prepared dispatch captures a specialization at bind time.
                binding = program.bind(inputs)
                actual, first_ms = _first_call(torch, lambda: binding.run(copy_outputs=False))
                entry["correctness"] = compare_arrays(_numpy_outputs(graph, actual), expected, case["dtype"])
                for _ in range(settings["warmup"]):
                    binding.run(copy_outputs=False)
                torch.cuda.synchronize()
                entry.update({"status": "ok", "first_use_wall_ms": first_ms,
                              "kernel_resources": [dict(dispatch.kernel_metadata) for dispatch in binding._dispatches],
                              "warm_cuda": _event_samples(torch, lambda: binding.run(copy_outputs=False), settings["repetitions"], settings["calls_per_sample"])})
            except Exception as exc:
                entry.update(_error(exc))
            results.append(entry)
    return results


def run_suite(output_path, quick=False, include_inductor=True):
    """Write raw results and raise after recording any mandatory validation failure."""
    settings = {"warmup": 5 if quick else 15, "repetitions": 15 if quick else 50,
                "calls_per_sample": 3 if quick else 10, "host_repetitions": 7 if quick else 20,
                "quick": bool(quick), "seed": 123, "include_inductor": bool(include_inductor)}
    report = {"schema_version": 1, "status": "running", "settings": settings, "metadata": _metadata(),
              "methodology": {"cuda_event_scope": "average per invocation, includes GPU stream gaps from host dispatch; not isolated profiler kernel duration",
                              "host_scope": "one invocation with synchronization before/after; includes Python and synchronization overhead",
                              "first_use": "first invocation in this process; may use existing Triton/Inductor disk caches; not guaranteed cold compile",
                              "memory": "warmed allocator peaks, resident inputs and custom arena included; driver, external allocations and compiler CPU memory excluded",
                              "ordering": "implementation order rotated by case; cases grouped by dtype and shape; laptop clocks, thermals and background activity remain confounders",
                              "inductor": "fullgraph=True, dynamic=False, default mode; no suppression or eager fallback; compiler in-memory state reset per case"},
              "cases": [], "tile_sweep": [], "failures": []}
    _write_report(output_path, report)
    try:
        from .gpu import ensure_available
        ensure_available()
        import torch
    except Exception as exc:
        report.update({"status": "unavailable", "backend_error": _error(exc)})
        _write_report(output_path, report)
        raise
    from .reference import execute
    from .torch_reference import make_callable
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["metadata"] = _metadata(torch)
    cases = benchmark_cases(quick)
    start_suite = time.perf_counter()
    try:
        with torch.inference_mode():
            for index, case in enumerate(cases):
                print(f"benchmark {index + 1}/{len(cases)} {case['id']}", flush=True)
                graph = case["graph"]
                entry = {k: v for k, v in case.items() if k != "graph"}
                entry["graph"] = graph.to_dict()
                entry["implementations"] = {}
                arrays = make_inputs(graph, seed=settings["seed"])
                expected = execute(graph, arrays)
                inputs = {name: torch.from_numpy(a).cuda() for name, a in arrays.items()}
                eager, args = make_callable(graph, inputs)
                eager_outputs, _ = _first_call(torch, lambda: eager(*args))
                framework_expected = _numpy_outputs(graph, eager_outputs)
                del eager_outputs
                try:
                    entry["framework_reference_check"] = compare_arrays(framework_expected, expected, case["dtype"])
                except Exception as exc:
                    entry["framework_reference_check"] = _error(exc)
                    report["failures"].append(f"{case['id']}: framework reference mismatch")
                names = ["custom_unoptimized", "custom_optimized", "pytorch_eager", "pytorch_inductor"]
                names = names[index % len(names):] + names[:index % len(names)]
                entry["measurement_order"] = names
                for name in names:
                    gc.collect()
                    if name == "pytorch_inductor" and not include_inductor:
                        entry["implementations"][name] = {"status": "disabled", "reason": "include_inductor=False explicitly requested"}
                        continue
                    try:
                        if name.startswith("custom_"):
                            result = _benchmark_custom(torch, graph, inputs, expected, framework_expected, case["dtype"], name == "custom_optimized", settings)
                        else:
                            fn = eager
                            wrapper_ms = 0.0
                            if name == "pytorch_inductor":
                                torch.compiler.reset()
                                wrapper_start = time.perf_counter()
                                fn = torch.compile(eager, backend="inductor", fullgraph=True, dynamic=False)
                                wrapper_ms = (time.perf_counter() - wrapper_start) * 1000
                            try:
                                result = _measure(torch, lambda: fn(*args), graph, expected, framework_expected, case["dtype"], settings)
                            except Exception as exc:
                                if name == "pytorch_inductor" and _optional_compile_error(exc):
                                    result = {**_error(exc), "status": "unavailable", "reason": "optional fullgraph Inductor compilation failed; no eager fallback"}
                                else:
                                    raise
                            result["compile_wrapper_ms"] = wrapper_ms
                            if result["status"] == "ok":
                                result["metrics"] = operation_metrics(graph, result["warm_cuda"]["median_ms"])
                                result["host_scope"] = "framework invocation includes framework output allocations; GPU inputs already resident"
                            del fn
                        entry["implementations"][name] = result
                    except Exception as exc:
                        entry["implementations"][name] = _error(exc)
                        report["failures"].append(f"{case['id']}: {name}: {type(exc).__name__}: {exc}")
                optimized = entry["implementations"]["custom_optimized"]
                if optimized["status"] == "ok":
                    target_ms = optimized["warm_cuda"]["median_ms"]
                    entry["speedup_vs"] = {name: result["warm_cuda"]["median_ms"] / target_ms
                                            for name, result in entry["implementations"].items() if result["status"] == "ok"}
                report["cases"].append(entry)
                _write_report(output_path, report)
            print("benchmark tile/warp sweep", flush=True)
            report["tile_sweep"] = _tile_sweep(torch, cases, settings)
            report["failures"].extend(f"tile sweep {x['case']} {x['config']}: {x['error']}" for x in report["tile_sweep"] if x["status"] != "ok")
        report["status"] = "failed" if report["failures"] else "ok"
    except Exception as exc:
        report["status"] = "failed"
        report["fatal_error"] = _error(exc)
        raise
    finally:
        report["elapsed_wall_seconds"] = time.perf_counter() - start_suite
        report["nvidia_smi_after"] = _nvidia_snapshot()
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        _write_report(output_path, report)
    if report["failures"]:
        raise ExecutionError(f"benchmark recorded {len(report['failures'])} failure(s); see {output_path}")
    return report


def profile_graph(output_path):
    """Profile baseline, optimized and framework MLP execution with real CUDA traces."""
    path = Path(output_path)
    report = {"schema_version": 1, "status": "running", "metadata": _metadata()}
    _write_report(path, report)
    try:
        from .gpu import ensure_available
        ensure_available()
        import torch
        from .runtime import Program
        from .torch_reference import make_callable
        graph = workload("mlp", m=128, k=256, n=384, dtype="float32")
        arrays = make_inputs(graph, seed=123)
        inputs = {name: torch.from_numpy(a).cuda() for name, a in arrays.items()}
        eager, args = make_callable(graph, inputs)
        baseline = Program(graph, backend="gpu", optimize=False).bind(inputs)
        optimized = Program(graph, backend="gpu", optimize=True).bind(inputs)
        functions = {"custom_unoptimized": lambda: baseline.run(copy_outputs=False),
                     "custom_optimized": lambda: optimized.run(copy_outputs=False),
                     "pytorch_eager": lambda: eager(*args)}
        with torch.inference_mode():
            for fn in functions.values():
                for _ in range(5):
                    fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=True) as profiler:
                for _ in range(5):
                    for name, fn in functions.items():
                        with torch.profiler.record_function(name):
                            fn()
                    torch.cuda.synchronize()
                    profiler.step()
        trace_path = path.with_suffix(".trace.json")
        table_path = path.with_suffix(".table.txt")
        profiler.export_chrome_trace(str(trace_path))
        table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=100)
        table_path.write_text("\n".join(line.rstrip() for line in table.splitlines()) + "\n", encoding="utf-8")
        averages = []
        for event in profiler.key_averages():
            averages.append({"name": event.key, "count": event.count, "self_cpu_time_us": event.self_cpu_time_total,
                             "total_cpu_time_us": event.cpu_time_total,
                             "total_device_time_us": getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0)),
                             "self_device_time_us": getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0))})
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        kernels = [event for event in trace.get("traceEvents", []) if event.get("cat") == "kernel"]
        report.update({"status": "ok" if kernels else "unavailable", "metadata": _metadata(torch), "graph": graph.to_dict(),
                       "trace_file": trace_path.name, "table_file": table_path.name, "operators": averages,
                       "gpu_kernel_events": len(kernels), "gpu_kernel_duration_us": sum(event.get("dur", 0) for event in kernels),
                       "kernel_names": sorted({event["name"] for event in kernels}), "profiled_repetitions": 5,
                       "kernel_resources": {"custom_unoptimized": [dict(d.kernel_metadata) for d in baseline._dispatches],
                                            "custom_optimized": [dict(d.kernel_metadata) for d in optimized._dispatches]},
                       "limitations": "Profiler overhead affects durations. Timeline shows launch/compute behavior; no Nsight occupancy, register, DRAM transaction or utilization counters are claimed."})
        _write_report(path, report)
        if not kernels:
            raise ExecutionError("profiler recorded no CUDA kernels; CUDA/CUPTI profiling is unavailable")
        return report
    except Exception as exc:
        if report["status"] != "unavailable":
            report["status"] = "failed"
        report["error"] = _error(exc)
        _write_report(path, report)
        raise
