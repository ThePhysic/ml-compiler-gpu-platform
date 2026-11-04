# Triton kernels and precision contract

The backend is a set of original kernels for this project's IR, rather than
calls to framework arithmetic. PyTorch supplies device tensors, allocation,
streams, and numerical baselines. Triton is imported only when GPU execution is
explicitly selected, so the compiler and simulator work without GPU packages.

## Mapping and engineering questions

| IR region | GPU mapping | Variable under study |
|---|---|---|
| add/mul/activation chain | A bounded SSA region in one memory pass; external broadcast indexing and register-resident intermediates | Memory traffic versus integer indexing and register pressure |
| softmax | One CTA per row; masked, centered FP32 max/exp/sum | Row width, tail padding, warp count, numerical stability |
| layernorm | One CTA per row; translated FP32 mean, centered variance and affine epilogue | Reduction width, large offsets, padding and variance stability |
| matmul | Configurable rectangular M/N/K tiles; FP32 accumulation | Tile dimensions, irregular dimension waste, warps, FP32 versus FP16 |
| matmul + bias + activation | The same dot loop and a register-resident epilogue | Launch count, eliminated writes, precision boundary preservation |
| sum | One CTA per output scalar, arbitrary single axis of a contiguous tensor | Coalesced final-axis versus strided non-final-axis loads |
| reshape/transpose | Materialized masked copy with explicit index remapping | Layout cost visible in the execution graph |

The elementwise kernel accepts a validated straight-line SSA region. The host
maps names to operand indices and passes immutable op/shape tuples to Triton's
static unrolling. This supports internal reuse (for example `(x+b)*s + (x+b)`)
and independent external broadcasts, rather than hard-coding a tutorial chain.
Compiler legality requires that all intermediate shapes equal the final shape.

Every graph tensor is contiguous and is an observable precision boundary.
FP16 matmul rounds its FP32 accumulator before bias addition, and rounds again
before an activation. Fused elementwise operations similarly convert every
intermediate back to the tensor dtype before the next operation. Removing an
allocation is not permission to change arithmetic semantics. Arithmetic kernels
disable FP fusion to avoid introducing FMA across separate FP32 operations.
The matmul dot explicitly selects `input_precision="ieee"` for float32 inputs;
host TF32 defaults cannot silently alter its precision. GELU uses the documented
project-wide tanh approximation, not the exact erf variant.
ReLU explicitly propagates NaNs to match the reference; Triton's default
`maximum` NaN policy would otherwise differ.
LayerNorm subtracts the first element of each row before reducing its mean. This
avoids a rounded sum of a large constant offset producing a false normalized value.
Tests include constant rows near 2^24 and independent FP64 oracles for variations
around large offsets. Epsilon must be positive, normal and finite in FP32;
subnormal values can flush to zero in GPU reciprocal-square-root arithmetic.

## Bounds and failure behavior

The public `launch(node, inputs, output, config=None)` API validates shapes,
dtype, device, contiguity, and output/input storage overlap even for independent
kernel callers. It writes into a supplied tensor and returns `None`; kernels
run asynchronously on the current CUDA stream.
Logical negative/conjugate views are rejected because their raw storage does not
represent the exposed tensor values. Prepared descriptors include those flags.

The runtime uses `prepare_launch` at binding time to validate the graph node
once and freeze all index/configuration metadata into a zero-argument dispatch
closure. Hot dispatch checks that tensor metadata and storage pointers remain
stable, without reconstructing IR. Inputs may receive new data between runs;
resizing or replacing their storage requires rebinding. Mutating the original
Node or configuration dictionary cannot change a prepared specialization.
After its first launch, the callable's `kernel_metadata` dictionary reports
Triton's measured compile metadata (registers per thread, spill slots, static
shared bytes, warp count and grid). These are resource descriptors, not a
measurement of achieved occupancy or utilization.

Supported dtypes are float16 and float32; rank is at most 8 and every tensor
contains fewer than 2^31 elements. Reductions support a width of 1..16384 and
normalize/reduce only the explicit supported axes. Softmax and LayerNorm use
the final axis; sum can use any one axis. LayerNorm weight/bias are `(width,)`.
Elementwise regions support 1..16 steps and at most 32 external tensors, enough
for every binary/unary region of that size. In particular a sixteen-add chain
can reference seventeen distinct graph inputs without losing fusion legality.
Matmul accepts only 2D `(M,K) @ (K,N)` and a fused bias of `(N,)` or `(1,N)`.
Activation is none, ReLU, or tanh GELU. Invalid cases produce explicit errors;
there is no arithmetic fallback.
The two-dimensional matmul launch also requires `ceil(N / block_n) <= 65535`,
CUDA's Y-grid bound. Larger grids raise `ResourceError` before launch; choose a
wider legal column tile or reduce the matrix width.

The default dot tile is 32x64x32 with 4 warps. Legal overrides use M/N tile
dimensions in {16,32,64,128}, K in {16,32,64}, and warps in {4,8}. A conservative
double-buffered operand footprint must fit the device's per-block shared
memory. This is a legality estimate, not an occupancy predictor. It does not
claim hardware autotuning. Reduction warps are 4 up to padded width 2048 and 8
above it. All non-power-of-two dimensions use masked loads and stores, including
the K tail of matrix multiplication.

`tests/test_gpu.py` covers both precisions, irregular dimensions, broadcast
indexing, SSA register reuse, exact FP16 fusion boundaries, configurable tiles,
constant normalization rows, large-offset softmax, non-final reductions,
materialized layouts, alias rejection, and explicit resource limits. GPU tests
skip only when optional packages or CUDA are unavailable; a CPU skip is not GPU
validation evidence. Run `python scripts/probe_gpu.py` before collecting data.

## Primary documentation used

- [Triton dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html): accumulator and IEEE input precision controls.
- [Triton JIT API](https://triton-lang.org/main/python-api/generated/triton.jit.html): specialization and compiled function restrictions.
- [Triton static_range API](https://triton-lang.org/main/python-api/generated/triton.language.static_range.html): compile-time unrolling for bounded SSA regions.
- [Triton sum API](https://triton-lang.org/main/python-api/generated/triton.language.sum.html): reduction dtype semantics.
- [Triton maximum API](https://triton-lang.org/main/python-api/generated/triton.language.maximum.html): explicit NaN propagation.
- [PyTorch CUDA availability](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.is_available.html) and [device context](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.device.html): explicit hardware detection and launch device selection.

Profiling and benchmarks must distinguish dispatch overhead from device latency
and compilation/JIT time. Speedups, bandwidth estimates, numerical tolerances,
and observed register/shared-memory metadata belong with the measured workload
and hardware. The kernel design alone does not establish a performance win.
