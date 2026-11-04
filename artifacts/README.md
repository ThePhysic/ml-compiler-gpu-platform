# Validation evidence

These files were generated on the actual RTX 4060 Laptop GPU in WSL2. The simulator
file is explicitly synthetic; it is not a GPU timing measurement.

| Artifact | Contents |
| --- | --- |
| `environment.json` | Actual Python, OS, dependency and GPU identity |
| `tests.xml` | Complete CPU/GPU test run; 270 passed, zero skipped |
| `demos/*.json` | Original IR, pass traces, execution/memory plans, instructions and backend correctness |
| `benchmarks.json` | 48 cases across four execution paths, 42 tile cases, raw timing/precision/memory data |
| `microbenchmarks.json` | 30 captured region comparisons and 42 tile captures |
| `microbenchmarks-repeat.json` | Independent repeat of the same captured experiment |
| `profile.json`, `.trace.json`, `.table.txt` | Real CUDA/CPU profile and kernel resource metadata |
| `simulator.json` | Functional correctness and synthetic estimates across three architectures |
| `summary.json` | Derived tables, reproducible with `scripts/summarize_results.py` |

All mandatory experiment statuses are `ok`; unavailable comparators are represented
explicitly by the tooling and must be checked when reproducing on another machine.
No CPU-only run is presented as GPU evidence. First-use timings may use disk caches.
Read [the technical analysis](../docs/results.md) before interpreting speedup ratios.

The ordinary benchmark/profiler/demos preceded the last validation-only input
hardening, which preserves generated kernels. CUDA Graph experiments and the final
test run include that hardening. Changes to analysis and evidence do not change
the measured kernel implementations. Raw reports are retained without filtering
slow samples or replacing results with their minimum.
