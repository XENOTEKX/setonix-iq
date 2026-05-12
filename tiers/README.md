# Thread-scaling audit — new runs (Tiers 1–3)

Generated from CHANGELOG entry `2026-05-12 (ae)`. These scripts add the missing
data points needed to make `tools/scaling_10M_analysis.py` Panel 1 scientifically
accurate (remove ⚠ warnings, anchor Amdahl T₁ for R2+NUMA / AVX, complete GCC
series, add MF-only baseline so MF2 ◆ is comparable).

## Walltime / KSU estimates

All wall times extrapolated from confirmed runs in the audit table. Walltimes
have ~30% safety margin. KSU = (ncpus × hours × 2.0) / 1000 on `normalsr`.

| Tier | Run                       | Binary               | THREADS | NRANKS | ncpus | Walltime | Est. wall | KSU    |
|------|---------------------------|----------------------|---------|--------|-------|----------|-----------|--------|
| 1    | R2+NUMA Clang 1T          | `build-profiling-clang` | 1       | 1      | 1     | 04:00:00 | ~3h20m    | 0.008  |
| 1    | R2+NUMA Clang 4T          | same                 | 4       | 1      | 4     | 02:00:00 | ~1h15m    | 0.016  |
| 1    | R2+NUMA Clang 8T          | same                 | 8       | 1      | 8     | 01:30:00 | ~45m      | 0.024  |
| 1    | R2+NUMA Clang 16T         | same                 | 16      | 1      | 16    | 01:00:00 | ~25m      | 0.032  |
| 1    | AVX-512+R2 OMP-only 1T    | `build-profiling-mpi` (um09 v3.1.2, np=1) | 1 | 1 | 1   | 04:00:00 | ~3h20m    | 0.008  |
| 1    | AVX-512+R2 OMP-only 4T    | same, np=1           | 4       | 1      | 4     | 02:00:00 | ~1h15m    | 0.016  |
| 1    | AVX-512+R2 OMP-only 8T    | same, np=1           | 8       | 1      | 8     | 01:30:00 | ~45m      | 0.024  |
| 2    | GCC Canonical 104T        | `build-profiling` (gcc)  | 104  | 1      | 104   | 01:00:00 | ~20–30m   | 0.208  |
| 3    | MF-only MF2 1T            | `build-mpi-mf2` (um09, np=1, AVX-512+R2+LPT) | 1 | 1   | 1     | 04:00:00 | ~1–3h     | 0.008  |
| 3    | MF-only MF2 4T            | same (np=1)          | 4       | 1      | 4     | 01:30:00 | ~20–40m   | 0.012  |
| 3    | MF-only MF2 8T            | same (np=1)          | 8       | 1      | 8     | 01:00:00 | ~10–20m   | 0.016  |
| 3    | MF-only MF2 16T           | same (np=1)          | 16      | 1      | 16    | 00:30:00 | ~6–12m    | 0.016  |
| 3    | MF-only MF2 32T           | same (np=1)          | 32      | 1      | 32    | 00:30:00 | ~3–6m     | 0.032  |
| 3    | MF-only MF2 64T           | same (np=1)          | 64      | 1      | 64    | 00:30:00 | ~2–4m     | 0.032  |
| 3    | MF-only MF2 104T          | same (np=1)          | 104     | 1      | 104   | 00:30:00 | ~1–2m     | 0.104  |
| 3    | MF2 dispatch 2-node 208T  | `build-mpi-mf2` (um09)   | 208  | 2 (1×104/node) | 208 | 00:30:00 | ~2m | 0.208 |
| **Total (Tiers 1+2+3)** |                |                      |         |        |       |          |           | **≈ 1.01 KSU** |

Tier 4 (ICX-compiled R2+NUMA isolation, 2 runs) requires a new binary build and is
intentionally deferred — see CHANGELOG (ae) Tier 4 for details.

### Why not "GCC 104T will waste KSU"?

Earlier concern was that 104T runs at full walltime would be expensive. In fact:
- 104T GCC walltime is short (~20–30 min from interpolation between 64T=1638s and ICX 104T=1112s)
- 1h walltime cap × 104 cpus × 2.0 = **208 SU = 0.208 KSU**

The truly long-running jobs (R2 1T ≈ 3h20m, AVX 1T ≈ 3h20m, MF-only 1T ≈ up to 3h)
reserve only **1 cpu** each, so each costs ≤ 8 SU = 0.008 KSU. They are essentially
free in SU terms despite being long in wall time.

## Layout

```
tiers/
├── README.md                       — this file
├── run_xlarge_mf_audit.sh          — MF-only MF2 runs (Tier 3 jobs 1–7, np=1, seed=1)
├── run_xlarge_avx_omp.sh           — AVX-binary in OMP-only mode (Tier 1 AVX jobs)
├── submit_tier1.sh                 — submits 7 anchor runs (R2+NUMA + AVX)
├── submit_tier2.sh                 — submits GCC 104T
├── submit_tier3.sh                 — submits MF-only series + MF2 2-node
└── submit_all.sh                   — submits Tiers 1+2+3 in dependency order (parallel within tier)
```

All record JSONs land in `logs/runs/`, where `tools/scaling_10M_analysis.py` will
pick them up automatically on the next run.

## Usage

```bash
# Submit one tier at a time (recommended — inspect output between tiers):
./tiers/submit_tier1.sh        # 7 anchor jobs, parallel
./tiers/submit_tier2.sh        # 1 job
./tiers/submit_tier3.sh        # 8 jobs

# Or fire all 16 jobs at once:
./tiers/submit_all.sh

# Dry-run (print the qsub commands without submitting):
DRY_RUN=1 ./tiers/submit_tier1.sh
```

After all jobs complete, regenerate the chart:

```bash
python3.11 tools/scaling_10M_analysis.py
```

Expected outcome:
- R2+NUMA Amdahl fit becomes reliable (loses `[⚠ T₁ extrap.]` flag)
- AVX-512+R2 Amdahl fit becomes reliable
- GCC family extends to 104T (shows NUMA penalty point)
- New 6th family “MF-only MF2” appears on Panel 1 — gives MF2 ◆ a comparable
  same-binary, same-protocol Amdahl curve to sit against (seed=1, -te fixed_xlarge_tree.nwk)
