"""Capture three real graph-to-GPU demonstrations from an installed checkout."""
import argparse
from pathlib import Path

from mlgpu.cli import main


parser = argparse.ArgumentParser()
parser.add_argument("--output-directory", default="artifacts/local/demos")
args = parser.parse_args()
directory = Path(args.output_directory)
for name, dtype in (("epilogue", "float32"), ("norm", "float32"), ("mlp", "float16")):
    status = main(["demo", "--backend", "all", "--workload", name, "--dtype", dtype,
                   "--output", str(directory / f"{name}.json")])
    if status:
        raise SystemExit(status)
