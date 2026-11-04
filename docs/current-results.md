# Current recorded results

The refreshed post-audit evidence shows useful fusion gains, matrix-kernel losses,
and a mixed integrated result on **one NVIDIA RTX 4060 Laptop GPU**. These are
recorded experiments, not new GPU measurements from the documentation polish.
The [original study](results.md) remains historical; the [audit](audit.md) explains
the subsequent correctness fixes.

## Hardware and source provenance

The measured environment was WSL2/Linux on a Ryzen 9 7940HS with an RTX 4060 Laptop
GPU: 8,188 MiB reported memory, 24 SMs, compute capability 8.9, NVIDIA driver 566.36,
CUDA 12.6 PyTorch wheels, Python 3.12.16, PyTorch 2.9.1+cu126, Triton 3.5.1 and
NumPy 2.2.6. FP32 framework matmul used `highest` precision with TF32 disabled.
See [environment](../artifacts/audit/environment.json) and the reports' `metadata`.

| Evidence | Source state | Recorded coverage |
| --- | --- | --- |
| [Runtime benchmarks](../artifacts/audit/benchmarks.json), [captured regions](../artifacts/audit/microbenchmarks.json), [profile](../artifacts/audit/profile.json) | `a3c34ea86c9d2c421bfaccf0103d8515a0219624`; `implementation_dirty=false` | 48 runtime cases, all four comparators successful; 30 captured region cases; 42 tile configurations in each suite |
| [Full tests](../artifacts/audit/tests.xml), [installed-wheel tests](../artifacts/audit/cpu-wheel-tests.xml), [installation record](../artifacts/audit/install.json) | Final correctness state `020c779eadf7cb55a42a4e9208a19293028e4d59` | 385 CPU/GPU tests, zero failures/errors/skips; separately, 237 CPU tests from an isolated installed wheel |

The later source correction restores caller precision state after measurement.
Kernel code and the precision used during those measurements are unchanged;
the performance runs still belong to `a3c34ea`, not the later commit. CPU test
success or deselected/skipped GPU cases cannot establish physical-GPU correctness.
The separate installed-wheel CPU check used Python 3.12.15, as recorded in
`install.json`; it is distinct from the Linux GPU measurement environment above.

## Device regions: measured gains and losses

These are **hot-cache, repeated fixed-input CUDA Graph device-region medians**
on that RTX 4060 Laptop GPU. Each graph contains 64 invocations; five replay
warmups precede 15 timed replays. Both custom Triton and PyTorch eager use this
capture policy. Times exclude JIT, ordinary host dispatch, allocation and transfers;
they are not user-facing end-to-end latency. [Timing protocol](benchmarking.md#cuda-graph-kernel-microbenchmarks).

The medium elementwise region is `GELU((x + bias) * scale)` on `(128,384)`;
medium matmul is `(128,256) @ (256,384)`.

| Region | Precision | Custom Triton median (us) | PyTorch eager median (us) | Ratio in this recorded run |
| --- | --- | ---: | ---: | --- |
| Fused elementwise | FP16 | 1.472 | 5.264 | custom 3.58x faster |
| Fused elementwise | FP32 | 3.472 | 11.728 | custom 3.38x faster |
| Matmul | FP16 | 7.104 | 3.647 | custom 1.95x slower |
| Matmul | FP32 | 85.808 | 24.080 | custom 3.56x slower |

Fusion removes intermediate materialization and combines three operations into
one kernel. That is consistent with the elementwise gain, but the comparison also
changes implementations; it does not isolate physical memory traffic as the cause.
The matmul losses remain visible rather than being replaced with a favorable tile.

Variability is material. The ranges below are sample p05-p95, not confidence intervals;
CV is sample standard deviation divided by the mean. All raw samples remain in
the [captured-region report](../artifacts/audit/microbenchmarks.json).

| Region / precision | Custom p05-p95 (us) | Eager p05-p95 (us) | CV custom / eager |
| --- | ---: | ---: | ---: |
| Elementwise FP16 | 1.410-1.744 | 5.135-6.208 | 0.093 / 0.079 |
| Elementwise FP32 | 3.408-4.555 | 11.606-12.152 | 0.128 / 0.016 |
| Matmul FP16 | 7.051-14.217 | 3.501-11.339 | 0.632 / 0.566 |
| Matmul FP32 | 85.650-136.310 | 24.000-24.518 | 0.191 / 0.012 |

## Integrated MLP: fewer launches does not guarantee a faster call

For the same GPU, the medium FP16 two-layer MLP uses input `(128,256)` and hidden
width 384. Optimization reduces **6 to 3 kernel launches** and planned arena extent
from **196,608 to 163,840 bytes**. Those bytes exclude external inputs/constants,
returned output copies and driver storage.

The following timings are **bound, synchronized host medians** with already-resident
GPU inputs. Custom runs reuse their bound arena and borrow outputs; framework calls
include their normal output allocations. Seven host samples follow warmup.
Compilation, binding and transfers are outside this boundary.

| Implementation | Host median (us) | Sample p05-p95 (us) | CV |
| --- | ---: | ---: | ---: |
| Custom unoptimized | 249.778 | 224.191-363.422 | 0.212 |
| Custom optimized | 330.320 | 287.310-450.216 | 0.192 |
| PyTorch eager | 263.264 | 195.120-1143.512 | 0.863 |
| PyTorch Inductor, full graph | 429.986 | 418.976-621.025 | 0.224 |

The optimized custom graph loses to eager and its own unoptimized baseline in these
host medians, despite the structural reductions; it has a lower median than Inductor
in this run. Python dispatch and kernel choices still matter, and the variability
does not support a universal ranking. The custom convenience API, which adds fresh
binding/allocation and owned output copies, has a separate median of **2,034.235 us**
for the optimized graph. It also excludes H2D/D2H transfers and is not an inference
or serving benchmark. [Runtime report](../artifacts/audit/benchmarks.json).

## Numerical accuracy

The runtime suite checks all implementations against both NumPy and PyTorch eager;
captured-region outputs are checked against NumPy after replay. FP16 comparisons use
`atol=0.01, rtol=0.01`; FP32 uses `atol=0.0002, rtol=0.001`. Shapes and dtypes must
match. Every mandatory recorded check passed. For the FP16 MLP above, the custom
optimized output's maximum absolute NumPy-reference difference is **0.0078125**;
the report retains per-output error and tolerance-fraction fields for every case.
The [385-test report](../artifacts/audit/tests.xml) adds operation-specific,
large-offset, aliasing, resource and failure-path checks beyond these benchmark inputs.

## GPU resource metadata and profiling

The selected custom region specializations report the following compiler resources:

| Region / precision | Registers/thread | Shared bytes | Warps | Launch grid |
| --- | ---: | ---: | ---: | --- |
| Elementwise FP16 | 14 | 0 | 4 | `(192,)` |
| Elementwise FP32 | 18 | 0 | 4 | `(192,)` |
| Matmul FP16 | 78 | 16,384 | 8 | `(2,3)` |
| Matmul FP32 | 80 | 24,576 | 8 | `(2,3)` |

The matrix grid offers six CTAs for 24 SMs, a visible static-parallelism limitation.
These descriptors do not measure achieved occupancy, utilization or DRAM transactions.
The [CUDA profile](../artifacts/audit/profile.json),
[timeline](../artifacts/audit/profile.trace.json) and
[operator table](../artifacts/audit/profile.table.txt) record 85 actual kernel events
across five warmed FP32 MLP repetitions per implementation. Profiler durations include
instrumentation effects and must not replace the unprofiled timings above.

## Simulated accelerator: synthetic costs only

The separate [simulator study](../artifacts/audit/simulator.json) functionally executes
`(37,41) @ (41,29) + bias + GELU` in FP32 on three modeled hardware configurations.
Every baseline and optimized result agrees with the numerical reference.

| Model | Baseline / optimized synthetic cycles | Baseline / optimized transferred bytes |
| --- | ---: | ---: |
| 64 KiB SRAM, default DMA | 3,676 / 956 | 73,804 / 24,164 |
| 2 KiB SRAM, default DMA | 3,676 / 2,919 | 73,804 / 56,636 |
| 64 KiB SRAM, slower DMA | 10,590 / 3,087 | 73,804 / 24,164 |

These are costs inside the documented [engineering model](simulator.md), not
measured GPU latency or physical-accelerator throughput. Smaller SRAM constrains
legal tile size; slower modeled DMA changes the transfer bottleneck.

## Trace the numbers

The selectors below identify unique objects by `id` in JSON arrays; they are
human-readable lookup notation, not literal JSONPath syntax. Every displayed time
is the named stored millisecond field multiplied by 1,000; ratios are computed from
unrounded medians. Rounded table values never replace stored samples.

| Displayed values | Committed artifact and field |
| --- | --- |
| Four region medians and variability rows | `artifacts/audit/microbenchmarks.json`: `cases[id="elementwise-medium-float16" or "elementwise-medium-float32" or "matmul-medium-float16" or "matmul-medium-float32"].implementations.{custom_optimized,pytorch_eager}.hot_cuda_graph_region.{median_ms,p05_ms,p95_ms,coefficient_of_variation}` |
| Faster/slower ratios | Same region fields: eager median / custom median for faster rows; custom / eager for slower rows; `custom_speedup_vs_eager` stores eager / custom |
| Region shapes, capture settings and resource table | Same selected cases: `dimensions`, `dtype`, `implementations.custom_optimized.{settings,kernel_resources[0]}`; resource columns are `registers_per_thread`, `shared_memory_bytes`, `num_warps`, `grid` |
| MLP host times and variability | `artifacts/audit/benchmarks.json`: `cases[id="mlp-medium-float16"].implementations.{custom_unoptimized,custom_optimized,pytorch_eager,pytorch_inductor}.warm_host_synchronized.{median_ms,p05_ms,p95_ms,coefficient_of_variation}` |
| MLP launches and arena bytes | Same MLP case: `implementations.{custom_unoptimized,custom_optimized}.{kernel_launches_per_run,arena_bytes}` |
| MLP convenience-call median | Same MLP case: `implementations.custom_optimized.end_to_end_host_synchronized.median_ms` |
| Accuracy example and tolerances | Same MLP case: `implementations.custom_optimized.correctness.numpy_reference.{tolerances,outputs.y.max_absolute_error}`; FP32 tolerance example: `microbenchmarks.json` case `elementwise-medium-float32`, `implementations.custom_optimized.correctness.tolerances` |
| Runtime coverage / comparator status | `artifacts/audit/benchmarks.json`: lengths of `cases` and `tile_sweep`, `failures`, `status`, and each `cases[].implementations.*.status`; captured coverage uses lengths of `cases` and `tile_sweep`, `status` and implementation statuses in `microbenchmarks.json` |
| Hardware and measured source | Benchmark and microbenchmark `metadata.{gpu,cpu,os,python,versions,nvidia_smi,precision,source}`; `artifacts/audit/environment.json` also records device and software identity |
| Final correctness counts/source | `artifacts/audit/{tests.xml,cpu-wheel-tests.xml}`: `testsuite` attributes `tests`, `failures`, `errors`, `skipped`; `artifacts/audit/install.json`: `source_commit`; see the [audit provenance](audit.md#validation) |
| Profile event/repetition counts | `artifacts/audit/profile.json`: `gpu_kernel_events`, `profiled_repetitions` |
| Synthetic cycles and bytes | `artifacts/audit/simulator.json`: `configurations.{default_64k,constrained_2k,slow_dma_64k}.{baseline,optimized}.estimate.{cycles,transfer_bytes}` |
| Reproduction completion | `artifacts/audit/verification.json`: `status`, `stages[].{command,exit_code,elapsed_seconds}` |

## Practical limits

This is a Python compiler with original typed IR/passes, Triton GPU lowering and an
independently executable instruction simulator. Evaluation covers static FP16/FP32
graphs on one consumer GPU. It does not establish arbitrary PyTorch import, dynamic
shapes, attention, autograd, production serving or end-to-end inference improvements.
Laptop clocks, power, background activity, fixed-input cache reuse and sample variance
limit generalization. First-use timings may reuse disk caches; algorithmic bandwidth
proxies are not physical hardware counters. The raw capture commands retain their
historical absolute paths; making future capture paths portable is a separate
follow-up, not a silent edit to these records.
