# Part 13 — Production Readiness & Branch Consolidation

**Status:** PLAN + audit findings. Execution gated on the CTF build/smoke job (172080357) passing.
**Date:** 2026-06-23
**Scope:** Get the GPU/JOLT/CTF source production-ready and collapse the GPU branch sprawl into exactly two branches — `gpu-kernel-prod` (clean production) and `gpu-kernel-dev` (keeps the CPU-parity test harness).

This document is the single source of truth for "what is left before production." It pairs the branch plan with the prioritised TODO list from the code audit (this session, fresh-eyes review agent + manual).

---

## 0. TL;DR

- The CTF + JOLT-default code is **functionally close** — algorithms are sound (use-after-free fixed, RNG isolated, self-tests compiled out of release, no stray TODO/assert in GPU paths). What remains is **hygiene and packaging, not deep correctness.**
- **3 blockers** before a clean `gpu-kernel-prod` can be cut: (B1) delete the 8 dev-only `*CrossCheckOnce` validators in the prod cut — **the `gpu_*_crosscheck` launchers are production, not dev; see §3.1**, (B2) guard CTF against partitioned/codon input + CPU-only builds, (B3) `.gitignore` + remove the `tc_decider` harness/binary and trim per-model validation log spew.
- **Branch sprawl:** the canonical source is `gpu-kernel-clean` (a strict superset of `gpu-kernel`). Collapse to `gpu-kernel-prod` + `gpu-kernel-dev`. The `gpu-modelfinder` scripts branch carries **5 unique commits that must be archived first.**
- Everything is protected by local `archive/*` tags already laid in both repos — nothing is deleted irreversibly.

---

## 1. Current branch inventory

### 1.1 Source code — `/scratch/rc29/as1708/iqtree3-gpu` → pushed to `XENOTEKX/setonix-iq` GitHub as `gpu-kernel*`

| Branch | HEAD | Date | Relationship | Disposition |
|---|---|---|---|---|
| `gpu-kernel-clean` | `cca7dbc1` | Jun 20 | **canonical**; 14 ahead of `gpu-kernel`, **0 behind** | → basis for **dev** (+ uncommitted CTF/polish) |
| `gpu-kernel` | `a972cefc` | Jun 14 | fully contained in `-clean` (0 unique commits) | **RETIRE** (archived) |
| `gpu-kernel-backup` | `6d7f7483` | Jun 14 | local only; diverged (15/15) old backup | **RETIRE** (archived, local delete) |

The working tree on `gpu-kernel-clean` has **11 modified tracked files** (the CTF + production-polish changes, uncommitted) + ~100 untracked job-output dirs.

### 1.2 Scripts / research / docs — `/home/272/as1708/setonix-iq` (`XENOTEKX/setonix-iq` GitHub)

| Branch | HEAD | Date | Relationship | Disposition |
|---|---|---|---|---|
| `gpu-modelfinder-src` | `0ec39066` | Jun 20 | current scripts/docs working branch | KEEP (canonical scripts) |
| `gpu-modelfinder` | `4e9ee2f9` | Jun 13 | **5 unique commits** not in `-src` | merge/cherry-pick → then RETIRE |

**The 5 unique commits on `gpu-modelfinder` (must preserve before retiring):**
- `26870137` G.6.2 DNA-1M -m MF CTF payoff PASS (F81+F+G4 == IQ-TREE BIC winner, 7.4–13×)
- `d22688f4` G.5.0 A100 beats np8 (1504→1355s)
- `0a938c35` CHANGELOG full -m MF parity table (AA + DNA)
- `eb7d0364` G.6 audit hardening + part9 §IX.10
- `4e9ee2f9` bench: A100 -m TEST CTF re-run (frozen_ab)

> Note: `gpu-modelfinder*` are **scripts/docs**, not kernel source. They are a *separate* axis from the `gpu-kernel-prod/dev` split and should **not** be folded into the `gpu-kernel-*` names. Consolidation recommendation in §3.3.

---

## 2. Target topology

```
SOURCE (iqtree3-gpu → GitHub gpu-kernel-prod / gpu-kernel-dev)
  gpu-kernel-dev    full tree; CPU-parity cross-check harness present,
                    compiled in only with -DJOLT_DEBUG_BUILD=ON.
                    This is where development + GPU-vs-CPU validation happens.
  gpu-kernel-prod   derived from dev with dev-only code physically removed:
                    no tc_decider, cross-check bodies stripped, debug logs trimmed.
                    Builds clean with -DIQTREE_GPU=ON (JOLT default-ON), no debug flag.

SCRIPTS (setonix-iq GitHub)
  gpu-modelfinder-src   canonical scripts/docs (after absorbing the 5 commits)
  (gpu-modelfinder retired → archive tag)
```

**prod vs dev is NOT hand-maintained as a permanent fork.** `dev` is the development branch; `prod` is *re-derived from* `dev` by a deterministic cleanup step (gate is already `JOLT_DEBUG_BUILD`; cleanup removes the dev-only files and gated blocks). This avoids drift: fix in dev → re-cut prod.

---

## 3. Production-readiness TODO (from the audit)

Severity: **BLOCKER** = must fix before cutting prod · **MAJOR** = fix before sharing · **MINOR** = polish.

### 3.1 Dev-only code that would ship into prod

- **[BLOCKER] Remove the dev cross-check methods in the prod cut.** The dev-only set is **exactly the 8 one-shot validators**, whose sole caller is the already-`JOLT_DEBUG_BUILD`-gated block at `tree/phylotree.cpp:1315`:
  - `tree/phylotreegpu.cpp`: `gpuLnLCrossCheckOnce` (:299), `gpuMixLnLCrossCheckOnce` (:331), `gpuDervCrossCheckOnce` (:786), `gpuMixDervCrossCheckOnce` (:844), `gpuMixAllBranchDervCrossCheckOnce` (:902), `gpuMixWeightEMCrossCheckOnce` (:979), `gpuMixJointOptimizeCrossCheckOnce` (:1089), `gpuFreeQGradCheckOnce` (:1289).
  - These sit in **two contiguous ranges** — `299–408` (the 2 lnL validators) and `786–1343` (the 6 derv/grad validators) — so the prod cut deletes those two blocks + the 8 declarations in `tree/phylotree.h` + the gated call site (`phylotree.cpp:1310–1316`). The dev branch keeps them (run under `-DJOLT_DEBUG_BUILD=ON`).
  - **⚠ DO NOT touch the `gpu_*_crosscheck*` launchers in `tree/gpu/gpu_lnl_intree.cu` or the `*CleanRoom*` helpers in `phylotreegpu.cpp`.** Call-graph trace (2026-06-23): `gpu_lnl_crosscheck`/`_mix`, `gpu_derv_crosscheck`/`_mix`, `gpu_allbranch_derv_crosscheck_mix` are invoked by `gpuComputeTreeLnLCleanRoom`/`...Mix` and `gpuComputeEdgeDervCleanRoom`/`...Mix`, which are the **live JOLT compute primitives** (`computeLikelihoodBranchGPU:410`, `computeLikelihoodFromBufferGPU:1361`, `computeLikelihoodDervGPU:1346`, `optimizeParametersJOLTMix:1751+`). The "crosscheck" name is historical (G.2/G.8 origin); they are **production**. Gating or deleting them would break the GPU likelihood/optimizer path. This corrects the earlier audit note that listed the 5 launchers as dev-only.
- **[BLOCKER] Remove the `tc_decider` standalone harness.** `tree/gpu/tc_decider.cu` (tracked, self-contained `main()`, not referenced by any CMake target) + `tree/gpu/tc_decider` (1.2 MB ELF, untracked, **not gitignored**). Dev-branch only; delete from prod; gitignore the binary now so a `git add` can't capture it.
- **[MINOR] Trim per-model validation log spew on the live path.** `tree/phylotreegpu.cpp:1642` `[JOLT] model=… GPU lnL … CPU lnL … rel=… PASS/MISMATCH` and `:1869` `[JOLTMIX] … rel=… PASS` print once per engaged model (~60 lines on a full `-m TESTONLY`), exposing internal GPU-vs-CPU validation framing. Keep a one-line "model X: GPU JOLT, N iters"; gate the `rel`/verdict behind `JOLT_DEBUG`. **Keep** `[GPU-BRANCH] active/fallback` (:414/:428), NaN→CPU and write-back-MISMATCH fallbacks (:1612/:1656) — those are real safety signals.
- **[OK] No stray `TODO`/`FIXME`/`assert` in GPU paths.** `ctfSelfTest` asserts are under `#ifndef NDEBUG` (compiled out of release). The env knobs (`JOLT_DEBUG`, `JOLT_NTILE`, `MIXJOINT_DBG`, `IQTREE_GPU_DIAG`, …) are zero-cost when unset — acceptable to keep as dev levers.

### 3.2 CTF correctness / safety (`main/phylotesting.cpp`)

- **[BLOCKER] Guard against partitioned/codon input.** `runCTFModelFinder` (:1461) is reachable for supertrees; it unconditionally builds a plain `Alignment`/`IQTree` from `iqtree.aln`. For `--ctf -p parts.nex` (SuperAlignment) or codon data this misbehaves/crashes. Fix: at entry, if `iqtree.isSuperTree()` or unsupported `seq_type`, warn and `return false` → clean fall-through to `runModelFinder`.
- **[MINOR] Disable `--ctf` in a CPU-only build.** `main/main.cpp:2304` (`#else`) warns for `params.gpu` but does not clear `params.ctf`, so `--ctf` still runs (slow CPU path, misleading "JOLT + CTF" banner). Fix: clear `params.ctf = params.jolt = false` in the non-`IQTREE_GPU` branch.
- **[MINOR] Leak-on-throw / exception safety.** `sub_aln` (:1521), `models_block` (:1492), per-iter `rtree` (:1613) are raw `new`/`delete`; all *normal* return paths free + restore params correctly, but a `std::bad_alloc`/CUDA throw mid-CTF leaks before `main`'s handler. Bounded (CLI `outError` calls `exit`), but wrap in try/catch or `unique_ptr` for robustness.
- **[MINOR] Stray artifact.** `.ctf_coarse.treefile` (:1537) written to the run prefix, never cleaned up.
- **[OK, verified] No use-after-free** (M1 fix correct: `coarse_tree` destructs before `delete sub_aln`; `~PhyloTree` doesn't own `aln`). Param restoration complete on all 3 returns. RNG isolated via private `ctf_rng`. Subsample clamped `min(ctf_subsample, nsite)`. `selectCTFTopK`/`ctfIneligible` skip-gate fires only on ineligible models (no silent selection corruption).

### 3.3 Build / repo hygiene

- **[MAJOR] `.gitignore` ignores none of the ~100 untracked job dirs.** The source-repo `.gitignore` covers only `build/`, `cmake-build-debug/`, `softwipe_build`. Polluting the tree: `build-gpu-on/`, `build-gpu-off/`, `bench_*/`, `ctf*/`, `*.gadi-pbs/`, `*.o<jobid>`, `frozen-binaries/`, `g[0-9]*_*/`, `prof_*/`, `smb_*/`, `tc_decider*`, `*.fa`. One `git add -A` would commit hundreds of MB. Fix: extend `.gitignore` before any commit.
- **[OK] Nothing that should be committed is untracked** — all real source changes are in the 11 modified tracked files.

---

## 4. Validation gates (pending — "implement all, then one validation")

1. **Build** — `build-gpu-on` (IQTREE_GPU=ON, JOLT_DEBUG_BUILD=OFF) **and** `build-gpu-off` (CPU parity) both compile clean. *(job 172080357, queued)*
2. **Banner** — `--ctf` → `Kernel:  JOLT + CTF` + `GPU:` line; `--jolt` → `JOLT`; `--no-jolt` → CPU SIMD banner.
3. **CTF end-to-end** — `--ctf` produces a "Best-fit model:" on a real alignment.
4. **CTF winner == wrapper winner** — native `--ctf` winner string == `run_ctf_*.sh`/`ctf_rerank.py` oracle (DNA-1M → F81+F+G4; AA → LG+G4), with lnL parity |Δ| < 1.0.
5. **CPU-parity unchanged** — `build-gpu-off` best model + lnL == frozen baseline; SIMD banner, no GPU block.
6. **CPU fallback intact** — an ineligible +R/+I refine logs the decline and completes on CPU.

---

## 5. Execution runbook (ordered; archive-first)

> Safety net already in place: local `archive/*` tags on both repos (see §6). Nothing below deletes an unarchived ref. **No remote (GitHub) branch deletion happens without explicit user confirmation** — that is outward-facing.

1. **Land prod-readiness fixes on `gpu-kernel-clean`** (B1 gate bodies, B2 CTF guards, B3 gitignore + tc_decider removal + log trim). *(after build smoke confirms the current code compiles)*
2. **Validate** (gates §4) on one GPU PBS pass.
3. **Commit** the CTF + polish + fixes to `gpu-kernel-clean`.
4. **Create `gpu-kernel-dev`** from `gpu-kernel-clean` (full tree, harness present, `JOLT_DEBUG_BUILD` gates it).
5. **Cut `gpu-kernel-prod`** from `gpu-kernel-dev` via the deterministic cleanup (remove `tc_decider.cu`, delete the `#ifdef JOLT_DEBUG_BUILD` cross-check blocks, drop the trimmed debug logs). Verify prod builds with `-DIQTREE_GPU=ON` only.
6. **Absorb the 5 `gpu-modelfinder` commits** into `gpu-modelfinder-src` (cherry-pick/merge), confirm, then retire `gpu-modelfinder`.
7. **Push** `gpu-kernel-prod` + `gpu-kernel-dev` to GitHub; **after confirmation**, delete the redundant remote branches (`gpu-kernel`, `gpu-modelfinder`).

---

## 6. Archive / rollback

Local tags laid this session (push to GitHub as the durable archive when ready):

**Source (iqtree3-gpu):** `archive/gpu-kernel-clean-cca7dbc1`, `archive/gpu-kernel-a972cefc`, `archive/gpu-kernel-backup-6d7f7483`
**Scripts (setonix-iq):** `archive/gpu-modelfinder-4e9ee2f9`, `archive/gpu-modelfinder-src-0ec39066`, `archive/gpu-kernel-a972cefc`, `archive/gpu-kernel-clean-cca7dbc1`

Any retired branch is fully recoverable from its `archive/*` tag.

---

## 7. Open decisions for the user

1. **Scope of consolidation** — source branches only (`gpu-kernel*`), or also tidy the scripts branches (`gpu-modelfinder*`)? *(Recommendation: do both, separately; scripts keep the `gpu-modelfinder-src` name.)*
2. **prod realization** — re-derive `prod` from `dev` by cleanup (recommended, no drift) vs. maintain both by hand.
3. **Remote deletes** — confirm before deleting `gpu-kernel` / `gpu-modelfinder` on the shared GitHub (archive tags pushed first).
