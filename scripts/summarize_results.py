"""Produce small derived tables; retain the original samples as primary evidence."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(directory):
    ordinary = read(directory / "benchmarks.json")
    captured = [read(directory / filename) for filename in ("microbenchmarks.json", "microbenchmarks-repeat.json")]
    summary = {"ordinary_status": ordinary["status"], "ordinary_case_count": len(ordinary["cases"]),
               "ordinary_tile_count": len(ordinary["tile_sweep"]), "ordinary_failures": ordinary["failures"],
               "captured_runs": [], "runtime_cases": []}
    for report in captured:
        rows = []
        for case in report["cases"]:
            custom = case["implementations"]["custom_optimized"]["hot_cuda_graph_region"]
            eager = case["implementations"]["pytorch_eager"]["hot_cuda_graph_region"]
            rows.append({"case": case["id"], "custom_us": custom["median_ms"] * 1000,
                         "eager_us": eager["median_ms"] * 1000, "eager_over_custom": case["custom_speedup_vs_eager"],
                         "custom_cv": custom["coefficient_of_variation"], "eager_cv": eager["coefficient_of_variation"]})
        summary["captured_runs"].append({"status": report["status"], "case_count": len(rows),
                                         "tile_count": len(report["tile_sweep"]), "rows": rows})
    for case in ordinary["cases"]:
        if case["workload"] not in {"epilogue", "norm", "mlp"} or case["shape_class"] != "medium":
            continue
        impl = case["implementations"]
        summary["runtime_cases"].append({"case": case["id"], "custom_unoptimized_bound_host_us": 1000 * impl["custom_unoptimized"]["warm_host_synchronized"]["median_ms"],
            "custom_optimized_bound_host_us": 1000 * impl["custom_optimized"]["warm_host_synchronized"]["median_ms"],
            "custom_owned_outputs_host_us": 1000 * impl["custom_optimized"]["end_to_end_host_synchronized"]["median_ms"],
            "eager_host_us": 1000 * impl["pytorch_eager"]["warm_host_synchronized"]["median_ms"],
            "inductor_host_us": 1000 * impl["pytorch_inductor"]["warm_host_synchronized"]["median_ms"] if impl["pytorch_inductor"]["status"] == "ok" else None,
            "baseline_arena_bytes": impl["custom_unoptimized"]["arena_bytes"], "optimized_arena_bytes": impl["custom_optimized"]["arena_bytes"],
            "baseline_launches": impl["custom_unoptimized"]["kernel_launches_per_run"], "optimized_launches": impl["custom_optimized"]["kernel_launches_per_run"]})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="artifacts")
    args = parser.parse_args()
    directory = Path(args.directory)
    result = summarize(directory)
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
