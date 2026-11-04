"""Reproduce estimated accelerator comparisons; no physical timing claims."""
import argparse
import json
from pathlib import Path

import numpy as np

from mlgpu.examples import make_inputs, workload
from mlgpu.reference import execute
from mlgpu.simulator import HardwareConfig, compile_simulator, estimate, simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/local/simulator.json")
    args = parser.parse_args()
    graph = workload("epilogue", m=37, k=41, n=29)
    inputs = make_inputs(graph, 123)
    expected = execute(graph, inputs)
    configs = {
        "default_64k": HardwareConfig(),
        "constrained_2k": HardwareConfig(local_memory_bytes=2048),
        "slow_dma_64k": HardwareConfig(bytes_per_cycle_per_transfer=8),
    }
    result = {"kind": "synthetic_estimates_not_physical_measurements", "graph": graph.to_dict(), "configurations": {}}
    for name, hardware in configs.items():
        entry = {"hardware": hardware.to_dict()}
        for label, optimize in (("baseline", False), ("optimized", True)):
            plan = compile_simulator(graph, hardware, optimize)
            actual = simulate(plan, inputs)
            for output in graph.outputs:
                np.testing.assert_allclose(actual[output], expected[output], atol=3e-4, rtol=3e-4)
            entry[label] = {"correct": True, "tiles": plan.tiles, "estimate": estimate(plan)}
        result["configurations"][name] = entry
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"simulator estimates written to {target}")


if __name__ == "__main__":
    main()
