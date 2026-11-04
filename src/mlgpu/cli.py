"""Reproducible command-line demonstrations and evaluation entry points."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .errors import PlatformError
from .examples import make_inputs, workload
from .reference import execute


WORKLOADS = ("epilogue", "norm", "mlp", "elementwise", "matmul", "softmax")


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def doctor(require_gpu=False):
    result = {"python": sys.version, "os": platform.platform(), "numpy": np.__version__}
    for package in ("torch", "triton", "mlgpu"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    try:
        from .gpu import ensure_available
        ensure_available()
        import torch
        device = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": device.name, "memory_bytes": device.total_memory, "capability": list(torch.cuda.get_device_capability(0)), "cuda_runtime": torch.version.cuda}
    except (ImportError, PlatformError) as exc:
        result["gpu"] = {"available": False, "reason": str(exc)}
        if require_gpu:
            raise
    return result


def demo(args):
    from .compiler import compile_graph
    from .runtime import Program
    from .simulator import compile_simulator, estimate, simulate
    graph = workload(args.workload, args.m, args.k, args.n, args.dtype)
    inputs = make_inputs(graph, args.seed)
    expected = execute(graph, inputs)
    optimized = not args.no_optimize
    plan = compile_graph(graph, optimize=optimized)
    report = {"workload": graph.name, "dtype": args.dtype, "shape_parameters": {"m": args.m, "k": args.k, "n": args.n},
              "input_ir": graph.to_dict(), "execution_plan": plan.to_dict(), "backends": {}}
    backends = ("reference", "simulator", "gpu") if args.backend == "all" else (args.backend,)
    for backend in backends:
        started = time.perf_counter()
        if backend == "simulator":
            simplan = compile_simulator(graph, optimize=optimized)
            actual = simulate(simplan, inputs)
            report["simulator_plan"] = simplan.to_dict()
            report["simulator_cost_estimate"] = estimate(simplan)
        elif backend == "gpu":
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            program = Program(graph, backend="gpu", optimize=optimized)
            device_inputs = {name: torch.from_numpy(value).to("cuda:0") for name, value in inputs.items()}
            actual_device = program.run(device_inputs)
            torch.cuda.synchronize()
            actual = {name: a.cpu().numpy() for name, a in actual_device.items()}
        else:
            actual = Program(graph, optimize=optimized).run(inputs)
        elapsed = (time.perf_counter() - started) * 1000
        atol, rtol = (0.02, 0.02) if args.dtype == "float16" else (0.0003, 0.0003)
        errors = {}
        for name in graph.outputs:
            np.testing.assert_allclose(actual[name], expected[name], atol=atol, rtol=rtol, err_msg=f"{backend} output {name}")
            errors[name] = float(np.max(np.abs(actual[name].astype(np.float32) - expected[name].astype(np.float32))))
        report["backends"][backend] = {"correct": True, "max_absolute_error": errors, "demonstration_wall_ms_including_setup": elapsed}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlgpu", description="Inspectable tensor compiler and GPU execution platform")
    sub = parser.add_subparsers(dest="command", required=True)
    doc = sub.add_parser("doctor", help="report environment; optional required real GPU probe")
    doc.add_argument("--require-gpu", action="store_true")
    doc.add_argument("--output")
    for command in ("demo", "inspect"):
        p = sub.add_parser(command)
        p.add_argument("--workload", choices=WORKLOADS, default="epilogue")
        p.add_argument("--dtype", choices=("float16", "float32"), default="float32")
        for dim, default in (("m", 17), ("k", 31), ("n", 47)):
            p.add_argument(f"--{dim}", type=int, default=default)
        p.add_argument("--no-optimize", action="store_true")
        p.add_argument("--output")
        if command == "demo":
            p.add_argument("--backend", choices=("reference", "simulator", "gpu", "all"), default="reference")
            p.add_argument("--seed", type=int, default=0)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--output", default="artifacts/local/benchmarks.json")
    bench.add_argument("--quick", action="store_true")
    bench.add_argument("--no-inductor", action="store_true")
    prof = sub.add_parser("profile")
    prof.add_argument("--output", default="artifacts/local/profile.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args.require_gpu)
        elif args.command == "inspect":
            from .compiler import compile_graph
            result = compile_graph(workload(args.workload, args.m, args.k, args.n, args.dtype), optimize=not args.no_optimize).to_dict()
        elif args.command == "demo":
            result = demo(args)
        elif args.command == "benchmark":
            from .benchmark import run_suite
            result = run_suite(args.output, quick=args.quick, include_inductor=not args.no_inductor)
        else:
            from .benchmark import profile_graph
            result = profile_graph(args.output)
        if getattr(args, "output", None):
            write_json(args.output, result)
        if args.command == "demo":
            print(json.dumps({"workload": result["workload"], "backends": result["backends"], "arena_bytes": result["execution_plan"]["memory"]["peak_bytes"], "output": args.output}, indent=2))
        elif args.command in {"benchmark", "profile"}:
            print(f"{args.command} evidence written to {args.output}")
        else:
            print(json.dumps(result, indent=2))
        return 0
    except (PlatformError, ImportError, AssertionError, ValueError, RuntimeError) as exc:
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
