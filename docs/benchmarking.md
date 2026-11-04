# Benchmark and profiling protocol

`benchmark.run_suite(output_path, quick=False, include_inductor=True)` measures a
real CUDA device. CPU execution and simulator estimates cannot substitute for
these results. `profile_graph(output_path)` exports a Chrome trace, operator table
and structured CUDA/CPU profiling summary. Neither function silently falls back
to another backend when the GPU is unavailable.

## Workload coverage

Both modes contain 48 cases: matrix multiplication, matmul+bias+GELU, fused
elementwise chains, normalization+residual+GELU, bias+softmax, MLP blocks,
independent LayerNorm and independent Softmax. Each has small, medium and
irregular shapes in FP16 and FP32. Quick dimensions `(M,K,N)` are `(16,32,64)`,
`(128,256,384)` and `(37,93,127)`. Full dimensions are `(32,64,96)`,
`(256,512,768)` and `(113,259,509)`. Inputs are reproducible seeded Gaussian
values with modest scale; dedicated kernel tests separately exercise difficult
numerical inputs and shape tails.

Every case compares custom unoptimized, custom optimized, PyTorch eager and
full-graph Inductor execution when requested and supported. Single-operation
cases isolate kernel interfaces. The optimized epilogue and elementwise cases
are single fused kernel regions; their original graph is the framework baseline.
Multi-operation cases measure compiler-to-runtime graph execution.

## Timing definitions

- **Compiler time:** custom IR validation, passes, scheduling, tiling and arena
  planning measured on the host, before binding or Triton compilation.
- **Bind time:** input validation, constants and arena allocation, plus a final
  device synchronization. Inputs have already been transferred to CUDA.
- **First use:** synchronized wall time of the first invocation in this process.
  It includes JIT or library initialization when needed. Existing Triton/Inductor
  disk caches may shorten this number; it is not a guaranteed cold compile time.
  Inductor wrapper creation is recorded separately, with actual deferred
  compilation included in first use. Its in-memory compiler state resets per case.
- **Warm CUDA:** CUDA events surround repeated invocations on the current stream;
  samples report average milliseconds per invocation. These intervals can include
  host-dispatch gaps between kernels, particularly for very small workloads.
  They are distinct from isolated kernel durations in the CUDA profiler trace.
- **Warm host:** one invocation with synchronization before and after, including
  Python dispatch and synchronization overhead. Custom execution uses a bound
  arena and borrowed outputs. Framework baselines include their normal output
  allocations. That allocation difference is intentional and explicitly reported.
- **End to end:** the custom convenience API validates resident GPU inputs,
  allocates a fresh arena, dispatches and clones owned outputs, then synchronizes.
  It excludes compilation and H2D/D2H transfers. Compare this against framework
  synchronized host time when allocation-inclusive behavior matters.

Quick mode uses 5 warmups, 15 CUDA samples with 3 invocations each and 7 host
samples. Full mode uses 15 warmups, 50 CUDA samples with 10 invocations each and
20 host samples. Both retain all samples, mean, median, p05, p95, extrema, sample
standard deviation and coefficient of variation. No confidence interval or
universal speedup is inferred from these finite measurements.

Implementation order rotates between cases to reduce a fixed-order advantage.
Cases remain grouped by dtype and shape. GPU clocks, laptop thermals, background
activity, power mode and warm caches can still bias results. Run on AC power,
close competing GPU workloads and repeat the experiment before generalizing.
The tool records power/thermal/clock snapshots without changing system settings.

## Precision and correctness

All implementations are checked against the NumPy reference and an independent
PyTorch eager result. FP32 uses `rtol=0.001, atol=0.0002`; FP16 uses
`rtol=0.01, atol=0.01`. The absolute tolerance covers cancellation near zero;
the relative tolerance covers accumulation and quantization differences. Shapes
and output dtypes must match, NaNs fail, and every result records maximum absolute
error and maximum fraction of the allowed tolerance. Benchmark inputs alone do
not prove robustness; operation-specific correctness tests remain required.

FP32 framework TF32 is disabled during the experiment and settings are restored
afterwards. Custom matmul requests IEEE input precision. Inductor uses
`fullgraph=True`, `dynamic=False` and default mode; compilation failures of this
optional baseline are recorded as unavailable with their error. There is no
eager fallback. Numerical/execution failures and custom backend failures make
the suite fail after writing its evidence. Explicitly disabling Inductor is
recorded separately and does not imply it was validated.

## Memory and algorithmic metrics

The compiler reports arena extent and naive aligned storage. Runtime measurements
use PyTorch allocator resident, peak allocated, incremental peak and peak reserved
bytes after warmup. Inputs and the current bound custom arena are included.
Driver memory, JIT/external allocations, and CPU compiler memory are excluded.
Cached reserved memory and live allocated memory are distinct; neither equals
the compiler's logical arena size.

For each plan, matrix multiply work is `sum(2*M*N*K)`. Effective GFLOP/s divides
that work by measured plan latency and excludes activation/reduction arithmetic.
The minimum external-byte proxy counts each declared input/constant once plus
each output once. Logical plan traffic counts each operation's inputs and output;
it ignores repeated matrix-tile loads, caching and physical transactions. These
byte counts divided by latency are **algorithmic bandwidth proxies**, not DRAM
bandwidth measurements or hardware utilization counters.

The tile sweep covers seven tile/warp configurations across all six matmul
shape/precision combinations. Every specialization is validated before timing;
first-use overhead is separate. Static compiler defaults can lose to another
configuration, which is useful evidence rather than a benchmark failure.
Custom records also retain the actual Triton-compiled kernel's registers per
thread, spill slots, shared-memory bytes, warp count and launch grid. These are
compiler resource statistics, not measured hardware occupancy or utilization.

## Evidence and profiling

Results record GPU name, compute capability, memory, SM count, CPU, OS, Python,
NumPy, PyTorch/Triton versions, CUDA build version, driver, power limit, clocks,
temperature, graph dimensions, dtypes, kernel configurations and timing policy.
The report is checkpointed after each case and preserves explicit failures.

The profiler records five warmed MLP repetitions for custom baseline, custom
optimized and PyTorch eager execution with labeled CPU scopes, CUDA activity,
shapes and allocator events. Inspect the exported trace for launch count and
stream gaps, and the table for device/CPU durations. Missing CUDA kernel events
produce an explicit unavailable result and failure. Profiler overhead affects
durations. Nsight occupancy, register pressure, hardware utilization and physical
DRAM counters are not claimed by these artifacts.

Primary API references: [CUDA events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html),
[allocator peak statistics](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html),
[PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html), and
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
