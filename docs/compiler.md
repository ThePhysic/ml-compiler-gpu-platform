# Compiler implementation

The compiler implements its graph transformations, scheduler, tile selector and
allocator directly in Python. It does not delegate graph compilation to FX,
Inductor, LLVM, MLIR or a framework compiler. Python keeps inspection and
experimentation simple; this project does not claim a C++ implementation.

## Pass pipeline and legality

`compile_graph` deep-copies and validates its input. Every transformation is
validated again and records its before/after IR, node counts and change flag.
An unoptimized plan retains all original operations in topological order.

1. **Constant folding** evaluates constant-only operations with the numerical
   reference semantics. A 64 MiB per-result cap bounds compile-time folding.
   Folded outputs become named constants. Unobservable nodes/constants are
   removed, while the original declared input signature remains unchanged.
2. **Simplification** bypasses internal identity operations, same-shape reshapes
   and identity transposes. Public graph output names are preserved. No algebraic
   `x * 0` or `x + 0` rules change NaN or signed-zero behavior.
   Existing fused regions receive scope-aware external operand substitution;
   colliding local definitions are renamed before aliases enter the region.
3. **Matmul epilogue fusion** recognizes a 2-D matmul with optional bias `(N,)`
   or `(1,N)` and optional ReLU/tanh-GELU. Each removed intermediate must have
   exactly one use and must not be a graph output. Fanout, repeated operands and
   full-matrix bias block the unsupported epilogue pattern. Fusion preserves the
   dtype rounding of the original matmul and bias intermediate, including FP16.
   A region claimed by one fusion cannot be claimed by another matmul producer;
   row-shaped matmul results can legally serve as each other's bias operands.
4. **Elementwise fusion** recognizes linear add/multiply/ReLU/GELU regions of
   at most 16 operations. Every intermediate output has the same shape and dtype
   as the region output, one use and no public-output obligation. External
   operands may use validated NumPy broadcasting. Each operation retains its
   own output-dtype rounding. This avoids broadcasting a materialized smaller
   intermediate into a larger region without representing that intermediate.
5. **Dead code elimination** retains only values reachable from graph outputs.

The fused IR stores its original elementwise SSA steps explicitly. Lowering
must honor those steps and cannot substitute arithmetic with different precision.
Before/after inspection records copy nested attributes, so later rewrites cannot
retroactively change an earlier pass's input graph.

## Dependency-aware scheduling

For every ready operation, the scheduler predicts the arena pressure from
allocating its output **before** releasing its last-use inputs. It selects the
operation with the smallest predicted peak, then smallest post-operation live
bytes, then original topological position. All byte counts use 256-byte aligned
reservations. The scheduler is deterministic and can move a reduction immediately
after a large producer to avoid keeping several large branch intermediates live.
It is a greedy heuristic, not a globally optimal schedule or GPU concurrency
model. A single CUDA stream executes its selected ordering.

## Memory planning and ownership

The allocator computes inclusive lifetimes `[producer_index,last_consumer_index]`.
An output cannot alias an input read by the same operation. Produced public
outputs remain live through the end of the plan. Inputs and constants remain
external; a public output that directly names either also remains external.

A best-fit free-list allocator splits ranges and coalesces adjacent released
ranges. Offsets and reserved lengths are aligned to 256 bytes; `Allocation.size`
is the actual tensor byte count, and `reserved_size` includes padding. Reuse is
allowed only when the previous lifetime ends strictly before the new producer.
`peak_bytes` is the allocated arena extent, including fragmentation and alignment,
not a sum of logical live values. `naive_bytes` is the sum of aligned reservations
without reuse. Both exclude inputs, constants, GPU driver/JIT allocations and
framework allocator caching. A caller-supplied byte budget applies to the actual
arena extent and fails explicitly if insufficient.

## Tile specialization

Matmul and fused matmul select among explicit `(M,N,K,warps)` candidates:
`(16,32,32,4)`, `(32,64,32,4)`, `(64,64,32,4)`, `(32,128,32,4)` and
`(64,128,32,8)`. A static score accounts for padded arithmetic, number of programs
and an accumulator-register pressure proxy. It selects legal shape-aware defaults;
it does not predict real GPU latency. Benchmark configuration sweeps provide
measured evidence and can expose where the heuristic regresses.

## Tests and limitations

The compiler tests verify rewrite equivalence in FP16/FP32, public names, fanout
barriers, bounded fusion, shape-changing broadcast barriers, scheduling effects,
inclusive lifetimes, disjoint concurrent intervals, reuse, alignment and explicit
budget failures. No importer, dynamic-shape specialization, multi-device scheduling,
automatic differentiation or optimal register allocation is claimed.
