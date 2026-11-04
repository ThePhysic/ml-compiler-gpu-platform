# Simulated tensor accelerator

This backend is an engineering model of a hypothetical tensor accelerator. Its
cycle counts are **synthetic estimates, never GPU measurements**. It models no
commercial architecture, cache hierarchy, GPU instruction set, thermal behavior,
or driver overhead. The functional interpreter uses NumPy on the CPU.

## Architecture and instruction set

The default machine has two identical tensor/vector compute units, one DMA unit,
64 KiB of SRAM split into eight independent 8 KiB banks, a nominal 1 GHz clock,
256 modeled floating-point operations per cycle per compute unit, and 32 bytes
per cycle per DMA unit. Each instruction incurs four setup cycles in addition to
`ceil(work / rate)`. All are explicit `HardwareConfig` assumptions, configurable
without changing lowering. Clock scaling only converts modeled cycles to time.

Four banks form one tile context; the number of active contexts is bounded by
both compute-unit count and `banks // 4`. Remainder banks may be unused. A matrix
context stores an A tile, B tile, FP32 accumulator, and optional bias tile in
separate banks. A generic tensor context packs operand tiles into its first
three banks and stores the result in its fourth. No bank may exceed its capacity;
the total of all bank contents cannot exceed SRAM. FP16 operands use two bytes,
while matrix accumulators always use four. Scratch banks are reused after their
tile's store completes. Full graph intermediates are kept in global storage;
`global_storage_bytes` deliberately reports this simpler simulator allocation,
not the compiler runtime's separately optimized arena.

The emitted instruction set is:

| Instruction | Numerical effect | Resource |
| --- | --- | --- |
| `load` | Global matrix/bias tile to an SRAM bank | DMA |
| `zero` | Initialize an FP32 accumulator tile | Compute |
| `mma` | FP32 tile accumulation of A × B | Compute |
| `epilogue` | Cast and optional bias/activation with semantic rounding | Compute |
| `store` | SRAM tile to a global matrix region | DMA |
| `load_group` | Gather a broadcast, layout, or reduction tile into SRAM | DMA |
| `vector` | Execute the specified bounded tensor instruction on that tile | Compute |
| `store_group` | SRAM result to a global vector/row region | DMA |

Generic instructions support the documented IR subset, including fused
elementwise regions, materialized reshape/transpose, single-axis sums, and
last-axis softmax/LayerNorm. They are explicit accelerator tensor instructions,
not an assertion that these operations have single-cycle implementations.
Normalization/softmax requires at least one complete reduction row in the local
context and fails clearly when it cannot fit. Sum tiling similarly retains the
entire selected reduction dimension for each output element. The matrix path
tiles the reduction dimension and can run with much smaller local memory.

## Lowering and scheduling

`compile_simulator(graph, hardware=None, optimize=True)` first applies the shared
compiler pipeline, then emits tensor instructions. Baseline matrix tiles use up
to 8 elements per dimension; optimized tiles use up to `max_tile` (32 by default).
Tile dimensions shrink to satisfy each SRAM bank's capacity. Generic output
tiles use up to 64 baseline or 256 optimized elements. Row tiles use up to one
baseline or four optimized rows, further bounded by SRAM.

The instruction DAG includes graph producer dependencies, accumulator chains,
and SRAM-bank reuse hazards. A deterministic global earliest-ready list
scheduler assigns compute and DMA units while reserving every accessed bank for
the instruction's duration. Separate contexts permit compute and DMA to overlap.
A consumer loads a tensor only after all producer tiles have been stored; the
model does not assume inter-operation tile pipelining. A validator checks
dependency completion, resource/bank exclusivity, bank capacities, SRAM value
identity, producer completion, and complete, non-overlapping output stores.

`simulate(plan, inputs)` executes this emitted stream in schedule order. Its
matrix multiplications use only loaded tiles, and its vector instructions use
only gathered local tiles. It never calls the whole-graph reference executor.
Inputs are checked for exact names, shape, dtype, and contiguous layout. FP16
matmul results are rounded before bias and again before activation, preserving
the original graph's materialization semantics. Tiled FP32 summation can differ
slightly from a full BLAS multiplication because reduction order differs.

## Cost assumptions and inspection

`estimate(plan)` reports modeled cycles/time, instruction count, transferred
bytes, modeled arithmetic work, peak resident SRAM bytes, global storage bytes,
and per-resource busy cycles/utilization. Matrix accumulation counts `2*M*N*K`
operations. Generic per-output counts are one for copies/simple elementwise,
20 for GELU, six for softmax, and 12 for LayerNorm. Sums count one per reduced
element. Epilogue accounting uses two per output for a non-GELU epilogue and 21
for GELU. These are deliberately simple cost weights: transcendental latency,
address-generation costs, gather transaction inefficiency, and register-level
temporary storage are not modeled. Global bandwidth and bank bandwidth have no
separate ceiling beyond DMA bandwidth and exclusive bank reservations.

The interpreter and estimator share the instruction format, not their execution
mechanism. Functional results validate arithmetic; modeled costs explain effects
within the stated assumptions. Baseline/optimized comparisons can show fewer
transfers and instructions from fusion and larger tiles without predicting any
particular RTX 4060 speedup. `SimulatorPlan.format()` emits the full graph,
hardware assumptions, chosen tiles, ordered node names, instruction dependencies,
resource assignments, bank reservations, and timeline as JSON.
