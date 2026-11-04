# Post-delivery correctness audit

See [current results](current-results.md) for a readable summary of the refreshed
measurements and their field-level provenance; [results.md](results.md) preserves
the historical study. The counts and GPU outcomes here are existing recorded
evidence, not a new hardware-validation run during documentation polish.

The second audit found real defects beyond the original passing suite and review.
The original 277-test result remains valid historical evidence; it did not cover
the cases below. The core fixes and 96 additional regressions are in source
commit `a3c34ea86c9d2c421bfaccf0103d8515a0219624`; a further 12 real-PyTorch
precision-state regressions and cleanup correction are in
`020c779eadf7cb55a42a4e9208a19293028e4d59` (108 added cases in total).

## Findings and fixes

| Finding | Reproduction / impact | Correction |
| --- | --- | --- |
| Overlapping matmul fusion | Two `(1,K) @ (K,N)` results merged by add can both claim that add as an epilogue, deleting a needed producer | Each region can be consumed by only one fusion; test both producer orders, activations and dtypes |
| Mutable inspection snapshots | Simplifying a fused region changes nested operand arrays inside earlier pass snapshots | Inspection serializes detached nested attributes; simulator tile dictionaries are detached too |
| LayerNorm cancellation | A constant FP32 row near 2^24 can normalize to approximately 1 instead of 0 in both the old reference and GPU kernel | Translate by the row anchor before mean reduction; compare with independent PyTorch constant-row results and FP64 variation oracles |
| Epsilon underflow | A positive subnormal epsilon can flush to zero during GPU reciprocal-square-root and produce NaNs | Require positive normal finite FP32 epsilon, and reject unrepresentable/overflowing values with GraphError |
| Logical GPU views | Contiguous negative-bit input/output views expose values different from their raw storage; old kernels silently used the wrong sign | Reject unresolved negative/conjugate views and include flags in prepared metadata guards |
| Bound source changes | A graph with no kernels can return a resized input without triggering per-kernel checks; unused inputs/constants have the same gap | Guard every source descriptor on every run, on CPU and GPU |
| Constant ownership | Mutating a borrowed CPU constant output can change another binding or the compiled program | Copy constants separately for each CPU binding, matching GPU ownership |
| Simulator declaration mismatch | Forged MMA operand banks can change A@B to B@B; declared bytes or scheduled operations can disagree with executed work | Derive and verify opcode operands, bytes, work, tile/depth, bank contents and dependencies against graph semantics |
| CUDA grid limit | A legal wide matrix and column tile can exceed the 65,535-program Y-grid limit and fail in a low-level launch | Detect the limit before launch and return ResourceError with a tile/shape remedy |
| Evaluation/error handling | Matching infinities can appear to pass the accuracy helper; malformed builder fields or mixed input keys can leak generic exceptions | Reject non-finite comparison values; recheck mutable IR fields and preserve subsystem error types |
| Precision side effects | The GPU demo changed the caller's TF32 setting, while profiling inherited potentially unfair settings | The custom demo leaves framework precision unchanged; profiling scopes and restores IEEE settings even on failure |

The simulator remains an engineering model. Its stricter validator adds explicit
matrix node/tile/depth/operand-role annotations without changing modeled instruction
costs or counts. Manually altered plans must obey that instruction schema.

## Validation

The expanded physical-GPU suite passed **385 tests**, with no skips or failures.
The CPU subset passed 237 tests. Additional automated exploratory checks covered 2,000
randomized compiler DAGs, shuffled orders, both precisions, branches, multiple
outputs and recompilation; simulator experiments covered 80 shape/dtype/hardware
combinations. Dedicated tests cover large-offset numerical oracles, logical views,
stale source metadata, simulator tampering, resource limits and snapshot isolation.

Current test and experiment files are in [artifacts/audit](../artifacts/audit/).
The refreshed benchmark reports include the source commit and whether implementation
files were modified at measurement time. The original study in [results.md](results.md)
and the original raw reports are preserved; their timings must not be attributed to
the updated LayerNorm kernel.
The refreshed measurements use `a3c34ea`; the subsequent precision-state cleanup
correction changes restoration after measurements, not kernel code or in-measurement
precision. The final tests and installed-wheel checks include that correction.

Build and installation checks include a source distribution and wheel, followed
by an isolated, non-editable CPU installation: all 237 CPU tests pass there too,
and installed Python modules match both the wheel and audited source byte for byte.
The dependency compatibility check passes. The refreshed 48-case runtime suite,
30-case captured suite, both 42-configuration tile sweeps, real CUDA profile and
all three backend demonstrations pass. Every eager/Inductor comparator is present
and successful. Hosted CI and final Astra medium review are recorded separately.

Astra's first post-audit round found the additional precision-state issue: restoring
the boolean TF32 flag after the named `medium` mode invalidated real PyTorch's
coupled backend state. Flags now restore first and the named mode restores last
in the profiler context and both benchmark entrypoints. Real-PyTorch tests cover
all three modes and exception exits, supplementing the earlier mock test.

The recorded automated adversarial model rereview used `gpt-6-astra` at medium
reasoning effort and reported **clean, with no remaining complaints** for source
`020c779`. Its separately delegated checks and defect history are recorded in
[review.md](review.md). This was not human review; the tests, regressions and
numerical oracles are the substantive correctness evidence.

Reproduce the hardware evidence in an installed CUDA environment:

```bash
python scripts/revalidate_audit.py --output-directory artifacts/local/audit
```

The numerical and resource boundaries remain explicit: static positive contiguous
FP16/FP32 tensors, bounded reductions/rank, one consumer NVIDIA GPU, and no automatic
framework fallback. The audit increases coverage; it does not prove the absence of
every possible defect or eliminate the documented performance variability.
