# Refreshed post-audit evidence

Implementation source: `a3c34ea86c9d2c421bfaccf0103d8515a0219624`.
Benchmark/profile metadata records this commit with `implementation_dirty=false`.

| Artifact | Verified result |
| --- | --- |
| `tests.xml` | 373 tests on WSL/physical RTX 4060; zero failures or skips |
| `cpu-wheel-tests.xml` | 237 CPU tests from isolated installed wheel; GPU cases intentionally deselected |
| `install.json` | Built-wheel checksum, installed-module path and byte-for-byte verification against source |
| `environment.json` | Physical GPU and measured Python/software environment |
| `benchmarks.json` | 48 cases, all four execution paths successful, 42 tile cases, zero failures |
| `microbenchmarks.json` | 30 captured custom/eager comparisons and 42 tile captures, all successful |
| `profile.*` | Real CPU/CUDA profile using scoped IEEE framework precision |
| `demos/*.json` | Correct graph execution on reference, simulator and GPU backends |
| `simulator.json` | Functional checks and synthetic estimates across three hardware configurations |
| `verification.json` | Reproduction commands, elapsed times and successful exit codes |

See [audit findings](../../docs/audit.md). These files describe the corrected
implementation, including translated-mean LayerNorm and stricter runtime checks.
The original study remains in the parent directory as historical evidence.
Performance variability and timing boundaries remain as documented; no new
universal speedup claims are made.
