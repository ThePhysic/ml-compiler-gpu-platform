# Acceptance map

Run commands from an installed checkout; GPU commands require the documented Linux
or WSL CUDA environment. Test success with GPU skips is CPU evidence only.

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Static typed SSA, dependencies, validation | `ir.py` | `tests/test_ir_reference.py` |
| Original graph transformations with legality | `compiler.py` | `tests/test_compiler.py` |
| Dependency scheduling, tiling, memory reuse | `compiler.py` | branch ordering and interval interference tests |
| Executed accelerator instructions and resource legality | `simulator.py` | `tests/test_simulator.py`, simulator demo artifacts |
| Distinct estimated resource/timing model | `HardwareConfig`, `estimate` | baseline/optimized simulator comparison |
| Original GPU implementations | `gpu/kernels.py`, `gpu/__init__.py` | `tests/test_gpu.py` on CUDA |
| Full graph GPU lowering, arena views, dispatch | `runtime.py` | `tests/test_gpu_runtime.py` on CUDA |
| Explicit unsupported/error cases | all layers | negative tests and nonzero CLI exit |
| Three independently runnable stories | `mlgpu` CLI | inspect, kernel benchmark/profile, demo --backend all |
| Numerical and performance evidence | benchmark module | checked-in raw JSON and results report |
| Private publication and clean Astra review | final git commit and review evidence | verified private remote head and explicit clean verdict |

The required demonstrations are `epilogue`, `norm`, and `mlp`. `softmax`,
`elementwise`, and isolated `matmul` provide extra kernel experiments.
Importer, attention, autograd, arbitrary dynamic shapes, and LLM-platform integration
are explicitly outside this implementation's core acceptance.

## Reproduce

```bash
mlgpu doctor --require-gpu
python -m pytest
mlgpu demo --backend all --workload epilogue --output artifacts/local/epilogue.json
mlgpu demo --backend all --workload norm --output artifacts/local/norm.json
mlgpu demo --backend all --workload mlp --dtype float16 --output artifacts/local/mlp.json
mlgpu benchmark --quick --output artifacts/local/benchmarks.json
mlgpu microbenchmark --output artifacts/local/microbenchmarks.json
mlgpu profile --output artifacts/local/profile.json
```

Committed evidence identifies the machine and versions used. See
[measured results](results.md) and the committed artifacts for the actual validation
outcome. Independent Astra review is recorded separately after its completion.
