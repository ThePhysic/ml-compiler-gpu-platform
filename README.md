# ML Compiler & GPU Execution Platform

An inspectable tensor compiler with original optimization passes, a resource-constrained
accelerator simulator, custom Triton kernels, and an integrated GPU runtime. It targets
small, static MLP and transformer-style fragments on one consumer NVIDIA GPU.

The compiler controls fusion, dependency scheduling, temporary-buffer reuse, matrix
tile specialization, and kernel dispatch. The simulator executes emitted instructions;
its synthetic timing model is separate from physical GPU measurements.

## CPU setup and demos

Python 3.10+ is supported; the tested GPU environment uses Python 3.12.

```bash
python -m venv .venv
# Linux/WSL: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
python -m pytest -m 'not gpu'
mlgpu inspect --workload epilogue --output artifacts/local/ir.json
mlgpu demo --backend reference --workload mlp
mlgpu demo --backend simulator --workload epilogue
```

## GPU setup

Use Linux or WSL2 with an NVIDIA driver and CUDA-enabled PyTorch. Native Windows
Triton is outside the supported environment. The project does not install drivers
or change power settings. The tested pins use CUDA 12.6 wheels compatible with the
development machine's driver; another machine must verify its driver compatibility.

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[test]'
mlgpu doctor --require-gpu
python -m pytest
mlgpu demo --backend all --workload epilogue
mlgpu demo --backend all --workload norm
mlgpu demo --backend all --workload mlp --dtype float16
mlgpu benchmark --quick --output artifacts/local/benchmarks.json
mlgpu profile --output artifacts/local/profile.json
```

For the exact measured dependency versions, use `requirements/gpu.lock` after
installing the CUDA PyTorch wheel from its explicit index. GPU tests skip when the
backend is unavailable; `doctor --require-gpu` and GPU demos fail instead. A CPU-only
test result cannot establish completion of GPU validation.

## Engineering demonstrations

1. **Compiler:** `inspect` shows typed IR, pass-by-pass before/after graphs,
   a dependency schedule, tile choices, inclusive lifetimes and arena offsets.
2. **GPU kernels:** independent kernel tests and benchmark tile sweeps expose
   numerical checks, launch configurations, latency and shape-dependent tradeoffs.
3. **Integrated execution:** `demo --backend all` executes the same graph on the
   reference, emitted-instruction simulator, and compiled Triton GPU path.

The initial subset includes 2D matmul, add/mul, ReLU/GELU, LayerNorm, Softmax,
restricted reshape/transpose, and one-axis reductions. There is no arbitrary
PyTorch graph import, autograd, distributed execution, or serving framework.
See [architecture and precision](docs/architecture.md) for exact semantics,
[compiler](docs/compiler.md), [simulator](docs/simulator.md),
[kernels](docs/kernels.md), and [benchmarking](docs/benchmarking.md).

Measured evidence and its interpretation belong in `artifacts/` and
`docs/results.md`. Performance claims must identify workload, precision, timing
boundary, baseline, and hardware; vendor-wide speedup claims are not a goal.
