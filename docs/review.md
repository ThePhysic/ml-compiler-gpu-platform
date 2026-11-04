# Automated adversarial model review history

Reviewer: `gpt-6-astra`, reasoning effort `medium`, separately delegated with the
project goal, source commit, validation evidence and explicit scope boundaries.
This was **automated adversarial model review, not independent human review**.
The model verdicts below record the process; reproducible regressions and recorded
test results provide the evidence. Later audits found defects after earlier clean
verdicts, so a model's verdict is not a correctness guarantee.

Start with the [current results](current-results.md), [385-test report](../artifacts/audit/tests.xml)
and [audit regressions](audit.md). The original delivery requested private GitHub
publication; references to that workflow below describe historical events rather
than a current publication requirement.

## Round 1 — commit 4901d2a

One P2 finding: simplification updated the outer input list of an existing fused
elementwise region without updating its internal external-operand references.
A valid identity followed by a preexisting fused region therefore failed compilation.

The fix substitutes operands using the region's own scope, renames local values
that collide with remapped externals, and avoids collisions with existing local
names. Regression cases cover both dtypes, local shadowing, transitive aliases,
duplicate remapped external operands, fresh-name collisions, and physical GPU
execution. The original source graph remains unchanged.

The separately delegated model ran the original 270-test suite on the physical
GPU: all passed. No other concrete complaints were reported in that round.
After the fix, 277 tests passed on the physical GPU, including the new regressions.
That revision was then submitted for another automated review before the initial
private GitHub delivery.

## Round 2 — commit dca0fbc844042f92e4ccf72e3285eb3d20cd3af6

Astra's explicit verdict: **CLEAN / NO REMAINING COMPLAINTS**.

The model reviewer confirmed that scoped substitution resolves the finding, including
local-name capture, transitive aliases, duplicate externals, and fresh-name
collisions. Its separate rerun of the complete WSL GPU suite recorded **277 passed in
11.44 seconds**, with zero skips or failures. Documentation and evidence provenance
were accepted by the model reviewer. It cleared that initial private delivery.

The delivery's final review-record change affected documentation only; its
implementation and validated kernel code remained the same at that stage.

## Post-delivery audit — first round, implementation a3c34ea

A fresh Astra medium model-review agent inspected the additional audit fixes and
refreshed evidence. Its separate rerun recorded 373 tests with no skips; it checked
the simulator contracts/source guards and tiny-variance LayerNorm against an FP64
oracle.

One remaining finding concerned PyTorch precision restoration: setting the boolean
TF32 flag after restoring the named `medium` mode corrupts its coupled backend
state. The mocked test did not expose the real library interaction.

## Post-delivery audit — rereview, source 020c779eadf7cb55a42a4e9208a19293028e4d59

The helper and both benchmark paths now restore flags before the named precision
mode. Twelve real-PyTorch regressions cover all three modes, normal and exception
exits, and both benchmark entrypoints. The original reproducer also passes.

Astra's explicit verdict: **CLEAN — NO REMAINING COMPLAINTS**. Its separate checks
verified all 12 new real-PyTorch cases and the original reproducer. The final full
physical-GPU suite run by the implementation agent passed **385 tests**, with no
skips or failures; the isolated installed wheel passed all 237 CPU cases.

The post-audit publication contains documentation/evidence updates after that
source commit. Refreshed performance reports identify their earlier execution
source because the final correction affects precision-state cleanup only.
