#!/usr/bin/env bash
# Run from any directory using an activated, CUDA-capable Python environment.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mlgpu doctor --require-gpu --output artifacts/local/environment.json
python -m pytest -p no:cacheprovider
python -m mlgpu demo --backend all --workload epilogue --output artifacts/local/epilogue.json
python -m mlgpu demo --backend all --workload norm --output artifacts/local/norm.json
python -m mlgpu demo --backend all --workload mlp --dtype float16 --output artifacts/local/mlp.json
python -m mlgpu benchmark --quick --output artifacts/local/benchmarks.json
python -m mlgpu microbenchmark --output artifacts/local/microbenchmarks.json
python -m mlgpu microbenchmark --output artifacts/local/microbenchmarks-repeat.json
python -m mlgpu profile --output artifacts/local/profile.json
python scripts/capture_simulator.py --output artifacts/local/simulator.json
python scripts/summarize_results.py --directory artifacts/local
