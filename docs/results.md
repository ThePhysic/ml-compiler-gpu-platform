# Measured results and engineering analysis

These experiments ran on November 3, 2026 (Los Angeles time), on an RTX 4060
Laptop GPU with 8,188 MiB reported memory, compute capability 8.9 and 24 SMs.
The host CPU is a Ryzen 9 7940HS. Execution used WSL2 Ubuntu, NVIDIA driver
566.36, Python 3.12.16, PyTorch 2.9.1+cu126, Triton 3.5.1 and NumPy 2.2.6.
FP32 framework TF32 was disabled. [Environment](../artifacts/environment.json)
and the experiment reports contain the measured software/hardware metadata.

## Correctness and coverage

The implementation test run passed **277 tests with no skips or failures** on
the physical GPU. [JUnit evidence](../artifacts/tests.xml) covers IR and compiler
algorithms, inclusive memory lifetimes, randomized branched DAGs, simulator
schedules/resources, independent GPU kernels, both precisions, shape tails,
numerical stability, binding ownership, errors, and full graphs.

The three required [demo artifacts](../artifacts/demos/) execute identical graphs
through NumPy, emitted accelerator instructions and compiler-selected GPU kernels:

| Graph | Precision | GPU maximum absolute difference from NumPy |
| --- | --- | ---: |
| matmul + bias + GELU | FP32 | 4.77e-7 |
| LayerNorm + residual + GELU | FP32 | 5.96e-7 |
| two-layer MLP + residual | FP16 | 0 in this demonstration |

The exact FP16 example is not a general bitwise-equivalence claim across devices
or accumulation orders. Each GPU test/experiment uses its documented tolerance.

## Device-region measurements

[Run 1](../artifacts/microbenchmarks.json) and
[run 2](../artifacts/microbenchmarks-repeat.json) each contain 30 region cases
and 42 matrix tile/warp configurations, all numerically validated. Each capture
contains 64 fixed-input repetitions; CUDA event replay time is divided by 64.
This amortizes Python launch gaps and measures hot-cache device regions. Eager
baselines use the same capture policy. It excludes validation, allocation, JIT,
transfers and ordinary host dispatch. See [methodology](benchmarking.md).

Selected medians below are microseconds per region. Full raw samples, p05/p95,
standard deviation and coefficient of variation remain in both reports.

| Region | Run 1 custom / eager | Run 2 custom / eager | Interpretation |
| --- | ---: | ---: | --- |
| elementwise, medium FP16 | 1.472 / 6.624 | 1.536 / 5.552 | lower custom median in both runs |
| elementwise, medium FP32 | 1.632 / 4.528 | 2.848 / 9.696 | lower custom median; absolute times vary |
| LayerNorm, medium FP32 | 1.792 / 3.216 | 3.728 / 6.256 | lower custom median in both runs |
| matmul, medium FP16 | 7.088 / 4.670 | 7.088 / 3.808 | custom loses in both runs |
| matmul, medium FP32 | 158.384 / 18.368 | 46.608 / 18.480 | substantial custom loss and variability |
| matmul+bias+GELU, medium FP16 | 8.222 / 7.504 | 8.383 / 7.328 | fusion does not overcome the vendor baseline here |

The elementwise region computes `(x+b)*scale` followed by GELU. Its lowering
keeps intermediates in registers with explicit dtype boundaries, reducing three
operations to one launch and eliminating intermediate tensor writes/reads.
The observed gains are consistent with those structural changes. They compare
different implementations as well as launch counts; no hardware-counter evidence
isolates memory traffic as the sole cause.

## Matrix tile tradeoffs and uncertainty

The default medium matmul configuration is 64x128x32 with eight warps. Its grid
has six CTAs for 24 SMs, limiting available parallelism to six active SMs for that
launch. Plain FP32 matmul reports 80 registers/thread, 24,576 shared bytes and
no spill slots. The fused epilogue reports 114 registers/thread with the same
shared footprint and grid. These are compiler metadata, not achieved occupancy.

In run 1 the default plain FP32 matmul samples are bimodal: seven are approximately
67–68 us and eight are approximately 158–414 us, yielding CV 0.660. The same tile
in the later sweep has median 71.440 us. A 32x64x64/four-warp candidate exposes
24 CTAs and has median 19.840 us in that sweep, with CV 0.547. This identifies an
underfill weakness in the static cost heuristic and a promising alternative.
It does not establish a portable speedup factor or justify replacing the default
on every GPU.

The faster fused-epilogue timing is a different specialization with more registers.
These artifacts do not establish a causal explanation for its large difference
from plain matmul. Suite-level clock/power/thermal snapshots cannot assign
individual outliers to clocks, thermals or WDDM scheduling. The repeated experiment
preserves these uncertainties rather than discarding slow samples. A target-SM
aware tile heuristic and controlled per-case telemetry are useful follow-up work.

## Integrated execution and memory

The [ordinary runtime suite](../artifacts/benchmarks.json) contains 48 graph/kernel
cases, successful eager and full-graph Inductor baselines for every case, and
42 tile configurations. It records compiler time, binding, first-use, warmed CUDA
and synchronized host timing, allocation-inclusive convenience calls, and
allocator peaks. All numerical checks passed.

Structural compiler effects are deterministic even when timings vary:

| Medium FP16 graph | Kernel launches before / after | Arena bytes before / after |
| --- | ---: | ---: |
| matmul + bias + GELU | 3 / 1 | 196,608 / 98,304 |
| LayerNorm + residual + GELU | 3 / 2 | 196,608 / 196,608 |
| MLP + residual | 6 / 3 | 196,608 / 163,840 |

The arena allocator already reuses dead storage in the baseline, so eliminating
an intermediate does not necessarily reduce arena extent. Inputs/constants,
returned output clones and allocator/driver storage are outside this metric.

For medium FP16 MLP, bound synchronized host medians were 248 us unoptimized,
282 us optimized, 200 us eager and 400 us Inductor in this run. The convenience
API, including fresh binding/allocation and owned outputs, measured 1,954 us.
For medium FP16 normalization, the custom bound median improved from 179 to
123 us, while eager measured 113 us. These show why fewer launches or a faster
isolated region do not guarantee a faster full invocation. Python validation,
specialization and allocation remain material costs for this research runtime.
No serving-performance or universal compiler-speedup claim follows.

The [CUDA profile](../artifacts/profile.json),
[timeline](../artifacts/profile.trace.json), and
[operator table](../artifacts/profile.table.txt) capture five warmed FP32 MLP
repetitions per implementation, including 85 actual GPU kernel events. The custom
schedule reduces six kernel launches to three per graph. Profiler overhead and
CPU annotations make those durations unsuitable as substitutes for unprofiled
measurements. The timeline and resource records support launch/bottleneck
inspection; no Nsight utilization, achieved occupancy or physical DRAM counter
measurements are claimed.

## Synthetic accelerator study

The [simulator comparison](../artifacts/simulator.json) executes
`37x41 @ 41x29 + bias + GELU` on three explicitly modeled configurations.
Every result agrees with the numerical reference. These estimates are synthetic
cycles, not measured RTX 4060 latency.

| Model | Baseline / optimized cycles | Baseline / optimized transferred bytes | Optimized local bytes |
| --- | ---: | ---: | ---: |
| 64 KiB SRAM, default DMA | 3,676 / 956 | 73,804 / 24,164 | 20,744 |
| 2 KiB SRAM, default DMA | 3,676 / 2,919 | 73,804 / 56,636 | 2,048 |
| 64 KiB SRAM, slower DMA | 10,590 / 3,087 | 73,804 / 24,164 | 20,164 |

On the default model, instruction count falls from 522 to 20. Smaller SRAM
forces smaller legal tiles and additional transfers. Slower DMA increases modeled
cost, exposing a different resource bottleneck. These conclusions follow from
the published model and assumptions, not calibrated commercial hardware behavior.

## Reproducibility and boundaries

`scripts/validate_gpu.sh` reproduces tests and experiments in an activated GPU
environment. `scripts/capture_demos.py`, `scripts/capture_simulator.py`, and
`scripts/summarize_results.py` reproduce their respective artifacts/tables. The
[derived summary](../artifacts/summary.json) retains case names linking to raw data.
Requirements lock files record the measured environment. Source, commit history
and evidence are published together after independent review.

Only one physical GPU was evaluated. Inputs/shapes are bounded, captures reuse
buffers, laptop power limits were not controlled, and some sample CVs exceed 1.
Repeated-run direction is evidence within this experiment, not a statistical
confidence bound. The static tile heuristic is inspectable and imperfect;
autotuning and architecture-specific occupancy modeling remain future work.
Graph import, attention, autograd, distributed execution and LLM integration remain
outside the completed core scope.
