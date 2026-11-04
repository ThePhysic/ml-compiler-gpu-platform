# Architecture and numerical contract

The Python implementation owns its SSA IR, rewrites, schedule heuristic, tile heuristic,
interval allocator, instruction lowering, simulator, and runtime. NumPy provides
numerical references; PyTorch provides GPU tensors, stream interoperability,
measurement, and independent framework baselines. Triton compiles custom GPU
programs. This is not a C++ or LLVM/MLIR implementation.

```
Graph -> validate/infer -> folding/simplification/fusion -> schedule
                                                     -> lifetime arena plan
                                                     -> Triton specializations -> GPU
                                                     -> tiled instructions -> simulator
Original graph -> independent NumPy/PyTorch numerical references
```

## Supported subset

Tensors have positive, static dimensions, rank >= 1, float16 or float32 dtype,
and contiguous row-major storage. Graph values are immutable SSA names.
There is no dtype promotion, implicit broadcasting beyond NumPy add/mul rules,
or automatic CPU/framework fallback. Inputs must have exact names, shapes,
dtypes, devices, and contiguous storage.

| Operation | Constraint |
| --- | --- |
| matmul | (M,K) @ (K,N); no batching |
| add, mul | broadcast-compatible static tensors |
| relu, gelu | GELU uses tanh approximation |
| layernorm | last axis; weight/bias vectors; finite positive eps |
| softmax | last axis; stable max-subtraction |
| sum | one integer axis, boolean keepdims; scalar outputs excluded |
| reshape | positive static target with equal element count |
| transpose | explicit permutation, materialized contiguous output |
| identity | explicit copy, removable internally |

Backend resource limits can be narrower than valid IR; they raise errors instead
of silently changing execution. See kernel and simulator documentation.

## Precision

Matmul accumulates in FP32 and materializes its result in the declared dtype.
Bias additions and every elementwise step round at their graph value boundary,
including inside fused kernels. Normalization and sums reduce in FP32; normalization
applies its affine transform in FP32 before casting. FP32 matmul requests IEEE
input precision rather than silently enabling TF32. FP32 accumulation order and
transcendental implementations differ across CPU/GPU, so equality is tolerance-based
across backends. Compiler rewrites on the CPU reference must preserve exact values.
Finite inputs are the validated numerical domain; NaN/Inf propagation is not a
cross-backend compatibility guarantee. No finite-input scan runs on every dispatch.

## Runtime ownership and timing

`Program` compiles a private graph copy. `bind` validates inputs, copies constants
to the target, and allocates one byte arena with typed contiguous views at planned
offsets. Inputs and constants stay external; pinned produced outputs live until
execution finishes. Distinct bindings own independent arenas. A binding is not
reentrant, and GPU dispatch uses the current CUDA stream. Callers must not reuse a
binding across concurrent streams or mutate inputs while operations are in flight.

`Program.run` includes binding and copied outputs. A bound execution's default
`run()` also copies outputs, allowing results to survive another invocation.
`run(copy_outputs=False)` returns borrowed views valid until the next invocation;
evaluation uses this mode to isolate preallocated execution. Cross-stream ownership
and host synchronization remain the caller's responsibility.

Compiler arena bytes exclude external inputs/constants, reference-computation
temporaries, returned output copies, GPU driver storage, and compiler caches.
CUDA allocator peak measurements and the plan's arena bytes measure different things.

See [compiler](compiler.md), [simulator](simulator.md), [kernels](kernels.md), and
[benchmarking](benchmarking.md) for algorithms, assumptions, and measurement methods.
