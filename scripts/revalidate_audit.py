"""Refresh post-audit evidence with an installed, CUDA-capable interpreter."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", default="artifacts/audit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = Path(args.output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "stages": []}
    commands = [
        ["-m", "pytest", "tests", "-p", "no:cacheprovider", "-q", f"--junitxml={directory / 'tests.xml'}"],
        ["-m", "mlgpu", "doctor", "--require-gpu", "--output", str(directory / "environment.json")],
        ["-m", "mlgpu", "benchmark", "--quick", "--output", str(directory / "benchmarks.json")],
        ["-m", "mlgpu", "microbenchmark", "--output", str(directory / "microbenchmarks.json")],
        ["-m", "mlgpu", "profile", "--output", str(directory / "profile.json")],
        ["scripts/capture_demos.py", "--output-directory", str(directory / "demos")],
        ["scripts/capture_simulator.py", "--output", str(directory / "simulator.json")],
    ]
    status = 0
    try:
        for command in commands:
            print("audit stage: " + " ".join(command), flush=True)
            start = time.perf_counter()
            completed = subprocess.run([sys.executable, *command], cwd=root)
            report["stages"].append({"command": command, "exit_code": completed.returncode,
                                      "elapsed_seconds": time.perf_counter() - start})
            if completed.returncode:
                status = completed.returncode
                break
        report["status"] = "ok" if status == 0 else "failed"
    finally:
        (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
