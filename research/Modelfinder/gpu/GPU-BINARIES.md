# GPU IQ-TREE3 binaries — what each one is, and when to use it

Single source of truth for the IQ-TREE3 GPU binaries on Gadi scratch. Written 2026-06-18 after
reorganising the binary tree, because the live build output and the frozen snapshots were sitting as
siblings in the same directory and kept getting confused for one another.

**Repo / source:** `/scratch/rc29/as1708/iqtree3-gpu/` — branch `gpu-kernel-clean`, HEAD `a972cefc`
(GitHub: `XENOTEKX/setonix-iq` → branch `gpu-kernel`, published trailer-stripped, tree byte-identical).

---

## The two kinds of binary — this is the whole point

| Kind | Location | Name source | On rebuild | Move/rename? |
|------|----------|-------------|------------|--------------|
| **LIVE** (cmake output) | `build-gpu-on/`, `build-gpu-off/` | cmake target name `iqtree3` | **regenerated** | **NO** — a rebuild recreates exactly `iqtree3`, and ~40 scripts hardcode the path |
| **FROZEN** (snapshot) | `frozen-binaries/` | hand-named, md5-tagged | untouched | yes — these are immutable copies you keep on purpose |

If you remember nothing else: **`build-*/iqtree3` is whatever you last compiled; `frozen-binaries/*` is a
specific historical binary that will never change.** Benchmarks that must be reproducible point at frozen;
development and current-feature runs point at live.

---

## The registry

| Use this binary | Path (under `/scratch/rc29/as1708/iqtree3-gpu/`) | md5 | What it is | Use it for |
|---|---|---|---|---|
| **LIVE GPU (current)** | `build-gpu-on/iqtree3` | `86fc5adf` | JOLT + GPU HEAD, commit `2277273d` (G.8.0/G.8.1a profile-mixture lnL + per-class). All current kernels: free-Q, +F, +G, +R coverage, **+ `k1_node_mix` mixture clean-room cross-check**. | Any new run, `--jolt --gpu`, all diagnostics, coverage audits, current-feature benchmarks. **Default.** (md5 changes every rebuild — this is the 2026-06-18 G.8.1a build.) |
| **LIVE CPU-only** | `build-gpu-off/iqtree3` | `ddde07c7` | Same source, GPU compiled OUT (`-DIQTREE_GPU=OFF`). | Proving the CPU path is byte-identical / unperturbed by the GPU port; pure-CPU baselines on a GPU-less node. |
| **FROZEN parity** | `frozen-binaries/iqtree3-g5.0-parity.b85d482f` | `b85d482f` | Optimisation-frozen at G.5.0 (PartB on-device reduction + kernel-fusion + base-sweep-skip + `d_theta`-reclaim). Was `build-gpu-on/iqtree3.frozen_ab`. | Reproducing the **published CTF numbers** (AA-1M / AA-10M `-m MF`). The 3 CTF scripts below pin this so the rows stay comparable. |
| **FROZEN archive** | `frozen-binaries/iqtree3-g5.0-fused.2ce44a8b.ARCHIVE` | `2ce44a8b` | Earlier G.5.0 fusion-validation freeze, **superseded** by the parity binary. Was `build-gpu-on/iqtree3.frozen_g50fuse`. | Nothing — orphan, 0 references. Kept for provenance; safe to delete. |

---

## Which scripts pin which binary

- **`frozen-binaries/iqtree3-g5.0-parity.b85d482f`** (the only frozen binary anything still uses):
  `run_ctf_1m_mf_energy.sh`, `run_ctf_10m_mf_aa_h200.sh`, `run_ctf_1m_test_a100.sh`.
  *(These reproduce the published parity rows; do not switch them to the live binary or the numbers stop being comparable.)*
- **`build-gpu-on/iqtree3`** (live): everything else — ~40 scripts, all diagnostics, coverage, bench sweeps, the eukaryote run.
- **`build-gpu-off/iqtree3`** (CPU-only): `run_g1_0_build_gpuvolta.sh` (build/parity check) and the G.1 build log.

## Historical names (for reading old CHANGELOG / job logs)

`CHANGELOG.md` and older logs refer to the parity binary by its old filename **`frozen_ab`** and the orphan
as **`frozen_g50fuse`**. These were *not* rewritten (history stays honest); md5 is the stable identifier:
- `frozen_ab` ≡ `b85d482f` ≡ now `frozen-binaries/iqtree3-g5.0-parity.b85d482f`
- `frozen_g50fuse` ≡ `2ce44a8b` ≡ now `frozen-binaries/iqtree3-g5.0-fused.2ce44a8b.ARCHIVE`

## Rebuild quick-reference

```bash
cd /scratch/rc29/as1708/iqtree3-gpu/build-gpu-on
cmake --build . -j --target iqtree3        # regenerates build-gpu-on/iqtree3 (live); frozen-binaries/ untouched
md5sum iqtree3                             # changes whenever source changes — that's expected
```

To freeze a new reproducibility snapshot, copy the live binary into `frozen-binaries/` with an md5-tagged
name and add a row above — never freeze in place inside `build-gpu-on/`.
