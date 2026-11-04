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
| Traceable validation and review history | source hashes, regressions and recorded reports | [385-test report](../artifacts/audit/tests.xml), [automated model-review history](review.md) |

The required demonstrations are `epilogue`, `norm`, and `mlp`. `softmax`,
`elementwise`, and isolated `matmul` provide extra kernel experiments.
Importer, attention, autograd, arbitrary dynamic shapes, and LLM-platform integration
are explicitly outside this implementation's core acceptance.

The initial delivery workflow included private GitHub publication. That is a
historical delivery event, not a current technical acceptance criterion for this
public repository. Source hashes and reproducible evidence identify what was
tested; automated model-review verdicts supplement those checks.

## CPU and GPU validation are separate

The [hosted CPU workflow](../.github/workflows/ci.yml) checks Python 3.10, 3.12 and
3.13 without GPU execution. The [GPU workflow](../.github/workflows/gpu.yml) is a
manual dispatch for an operator-provisioned self-hosted CUDA runner. The committed
physical-GPU evidence comes from the recorded RTX 4060 Laptop GPU / WSL2 runs;
hosted CPU CI and skipped GPU tests do not replace it.

## Reproduce

CPU inspection and correctness checks:

```bash
python -m pytest -m 'not gpu'
mlgpu inspect --workload epilogue
```

In a configured NVIDIA Linux/WSL2 environment, the following commands reproduce
GPU demonstrations and experiments; they do not describe new runs made during
documentation polish:

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
[current results](current-results.md), the [historical study](results.md), and the
committed artifacts for recorded outcomes. The [automated adversarial model-review
history](review.md) retains defects, fixes and retests without implying human review.
