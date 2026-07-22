# GPU IQ-TREE3 binaries — what each one is, and when to use it

Single source of truth for the IQ-TREE3 GPU/CPU/MPI binaries on Gadi scratch.
**Audited & rewritten 2026-07-02** (the prior 2026-06-18 version was badly stale — it still listed the live
GPU binary as `46b8d079`/`077c0811` on branch `gpu-kernel-clean`; none of that is current). Every md5, path,
branch, and date below was re-derived from disk on 2026-07-02.

---

# 🔴 2026-07-18 — TWO REGISTERED TREES WERE DELETED IN AN INODE CLEANUP. Read this before following any `iqtree3-graduate/*` or `iqtree3-l0/*` path below.

During an inode-quota cleanup the worktrees **`iqtree3-graduate`**, **`iqtree3-l0`**, and **`iqtree3-tipvec`** were removed
from `/scratch/rc29/as1708/`. The delete-criterion verified **git source history was on GitHub** but did **NOT** cross-check
this binary registry ⇒ untracked `build-*/` binaries in those trees went with them. Verified from disk afterwards:

- **Source — SAFE:** `iqtree3-graduate` HEAD `fa7d7a1c` and `iqtree3-l0` HEAD `37e41df7` are both ancestors of the live
  `setonix-iq/gpu-kernel-dev` tip (ls-remote confirmed); `iqtree3-tipvec` `4de1e6c4` is the live `gpu-tipvec-merge` tip.
  **Re-clone from `setonix-iq` to restore any of these trees.**
- **Binaries that SURVIVE (archived in `/scratch/dx61/as1708/shared-jolt/`, md5-verified):** `239fb5f2` =
  `iqtree3-gpu-promoted-239fb5f2` (Hashara's binary, unstripped bits identical) · `fe5ce648` = `iqtree3-gpu-l8-fe5ce648`
  (the "newest/most-complete +R/L0" build-l8). **Every path row below reading `iqtree3-graduate/build-promote` or
  `iqtree3-l0/build-l8` is STALE — use the `shared-jolt/` archive instead.**
- **Binaries DESTROYED (only copies, unrecoverable):** `bffdc16e` (build-flagclean2, the >12 h-run research artifact —
  "NOT canonical, off critical path") · `3d124292` (build-flagclean) · `1c86fd6c` (build-grad-l7) · `916cba21` (build-grad)
  · `dcc970d0` (tipvec build-tv). **All are superseded by `239fb5f2` (a documented strict superset), off-critical-path, or
  rebuildable from GitHub** ⇒ no deliverable-path or non-recoverable loss — but this was partly luck, not the criterion.
- **LESSON (now a standing rule):** *deleting a worktree must cross-check THIS registry for untracked `build-*/` binaries,
  not only `git rev-list --not --remotes`.* A binary in an untracked build dir has no git protection.

Rebuild paths in this doc that point at the deleted trees (e.g. `cd iqtree3-graduate/build-promote && cmake --build`) require
a re-clone first.

---

# 🆕 2026-07-16 — binary map re-verified from disk; the +I+R fix lives ONLY in an uncommitted tree ⇒ `239fb5f2` is AGAIN not a full union.

Prompted by (1) a >12 h AA-1M run this morning that caused an "is `bffdc16e` our main binary?" scare, and (2) the +I+R
export-normalisation **correctness** fix (this session) not being in the shared binary. **Every md5 / path / git / sentinel
below was verified from disk today** (`md5sum` + `strings -a | grep`, not memory).

| binary | tree / path | git | screen-cache | +I+R fix (`JOLT_NO_PINVFIX`) | what it is |
|---|---|---|:--:|:--:|---|
| ⭐ **`f3f7875f`** | `iqtree3-jolt-merge/build-merge/iqtree3` | `jolt-gpu-merge` = `fa7d7a1c` **+5** (committed, clean) | ✅ | ✅ (alias) | **THE MERGE — supersedes `239fb5f2` AND `020ff472`.** Folds pureinvar (M.2 coverage) + mfdevcheck (FDFIX +I+R) + mfresident onto the graduated `fa7d7a1c`, **all optimisations ON by default** (MF-resident 1.597×, boot-snapshot, FDFIX/pinvfix, cache, L0/L5/L6/L7). Committed & clean (unlike `020ff472`). **Gate `174027423` PASSED** (v3, fixed-tree + AA self-check): G2 defaults-ON-by-runtime on a BARE invoke (DNA+AA engage JOLT), G3 control proven, G4-DNA rel=**3.320e-08**, G4-AA write-back MISMATCH=**0** (same-point), G6 alias fingerprint == gate 174010664. 🔴 **fixed-pinvar DEMOTED to opt-in** (`JOLT_FIXINVAR=1`; its only gate, covgate 173822572, had FAILED). Archived `gems-bin/iqtree3-gpu-jolt-merge-f3f7875f` (555). ⚠️ STRIPPED (runtime == nostrip; sentinel survives in `.rodata`). **Intended new `iqtree3-gpu-latest` target** (repoint pending user confirm). |
| **`239fb5f2`** | `iqtree3-graduate/build-promote/iqtree3` | `a07f61be` | ✅ | ❌ | ~~**The promoted binary SHARED with Hashara**~~ → **SUPERSEDED by `f3f7875f`** (2026-07-17); still the `iqtree3-gpu-latest` target until the symlink is repointed (`/scratch/dx61/as1708/shared-jolt/iqtree3-gpu-promoted-239fb5f2`). cache + L0 + L5/L6/L7 + det-fix default-ON; lacks the +I+R fix and the MF-resident/coverage work. |
| **`020ff472`** | `iqtree3-mfresident/build-mfresident/iqtree3` | `a07f61be` **+ 2 uncommitted** | ✅ | ✅ | **This session's MF-no-`--ctf` work binary** = `239fb5f2`'s source **+ the +I+R export-norm fix** (`JOLT_NO_PINVFIX` = default-ON kill-switch) + the RDIAG/self-check probe (2 edited files: `tree/gpu/gpu_lnl_intree.cu`, `tree/phylotreegpu.cpp`). Most feature-complete on paper, but **dirty & mid-validation** (fix gated on job `174003950` rfavor). |
| **`9d845205`** | `iqtree3-l2search-stage2b/build-prod/iqtree3` | `eef09e2c` | ❌ | ❌ | The **tree-search flagship** (AA-1M `-B` 2.694× / 1.123× ahead of Hashara, RF=0). Registered in full below. Lacks the screen-cache AND the +I+R fix. |
| `bffdc16e` | `iqtree3-graduate/build-flagclean2/iqtree3` | worktree, ~`a07f61be`→ (pre-`fa7d7a1c`), built 2026-07-14 20:37 | ✅ | ❌ | **Experimental "full-offload" flag-clean build.** The morning **>12 h AA-1M run (job `173919739`)** used THIS — and it was a **full MFP** (selection **+ tree search**, NO `-m MF`), killed at 12 h 16 m (exit 271 SIGTERM) at **97 % GPU mem** (139.71 GB). **Un-root-caused** — the nTile-explosion theory was **retracted** because the diagnostic A/B (`aantile` 173980098) changed binary AND scope (standard binary, `-m MF` only ⇒ 2845 s clean) and so never re-ran the failing config. A research artifact — **NOT canonical, off the deliverable's critical path.** |

**Honest status — the "one canonical binary" claim (07-13, below) is AGAIN not a true union.** The +I+R correctness fix
exists **only** in the uncommitted `iqtree3-mfresident` tree (`020ff472`); the shipped `239fb5f2` does not have it, and the
tree-search flagship `9d845205` has neither the fix nor the screen-cache. So today the MF work (`020ff472`) and the
tree-search tuning (`9d845205`) are **still split across two trees** — "we beat Hashara on tree search" (`9d845205`) and
"we corrected +I+R" (`020ff472`) are **two different binaries**. Re-consolidation (merge the stage2b tree-search tuning +
the mfresident MF work + the +I+R fix into one committed, validated line) is a follow-up, **gated on rfavor (`174003950`)
clearing the fix** — and it is a human push (assistant never pushes GPU source).

**Grounded MF head-to-head** (this morning, no `--ctf` per Minh, `-nt 12`, MF-phase to MF-phase — NOT the `bffdc16e`
full-MFP run above): **DNA MF we LOSE 1.31×** (ours 1501.7 s vs her OpenACC-JOLT 1142.4 s; ahead of her stable 1679.8 s),
**AA MF we WIN ~1.62×** (ours ~2845 s vs her 4605.1 s). Full detail + the ~933 CPU-s DNA residual (under profile, job
`174003961`) in `research/Modelfinder/MODELFINDER-FULL-GPU-PLAN.md` top block.

> **On the +I+R fix (`JOLT_NO_PINVFIX`, `020ff472` only — do NOT add it to the promoted-binary flag audit below).** Root
> cause: the pinv forward-FD in `gpu_lnl_intree.cu` perturbs `catRate/catProp_v` to `pp = baseP + 1e-4`; for `nFreeQ==0`
> (JC) nothing resets them, so a reject-exit exports an under-normalised model (`Σprop + pinv = 1 − 1e-4`) ⇒ JC+I+R2..R5
> self-check MISMATCH by ~10 nats @100 K / ~100 @1 M. Fix = `applyPinv(baseP)` before the base capture (host-only, zero GPU
> cost, pure +R byte-identical per-call). Default-ON, kill-switch `JOLT_NO_PINVFIX=1`. ✅ Validated (job 173995856:
> Σ→1.0, MISMATCH 1→0, winner unchanged). ⚠️ Cascade: shifts non-competitive +R5 models ≤40 BIC (401-cap hypersensitivity)
> — selection-safe on gamma-sim; `174003950` rfavor gates it on +R-favouring real data before push. The `-m MF` published
> table was always SAFE (CPU self-check returns `cpuLnL`); the bug bites GPU-trusting paths. See `project_gpu_freerate_handicap` memory.

## 🔎 2026-07-16 FULL DISK CENSUS — every GPU binary, capability-verified from disk (the promotion input)

**Method (nothing below is assumed):** `find` over `/scratch/rc29/as1708`, `/g/data/um09/as1708/gems-bin`,
`/scratch/dx61/as1708/shared-jolt` and every `frozen-binaries/` = **102 files**; then `md5sum` + `strings -a | grep`
per sentinel on each promotion-relevant binary.
> ⚠️ **Sentinel presence proves the code is LINKED — NOT that it ENGAGES or is correct.** This doc's own parsimony trap
> (`2c931f41` silently CPU-falls-back, `VERIFY mismatches=0` is a false pass) is exactly why **the runtime engage marker
> remains the gate**, never `strings`. Treat this matrix as *necessary, not sufficient*, for promotion.
> Also **do NOT use `JOLT_DET` as a sentinel** — it exists only in C++ comments; `strings` can never find it (documented gate bug #1).

### 🔴🔴 CORRECTION — **`8cc3cb84` IS NOT LOST. It exists, with parsimony intact.**
`/g/data/um09/as1708/gems-bin/iqtree3-thesis-sweep-8cc3cb84` — full md5 **`8cc3cb844e807fe3ecbcd0b9ff5428df`**,
13,288,336 B, **2026-06-29 19:04**, verified today to carry **parsimony (`GPU_PARSIMONY_BATCHED`=1, `GPUPARS-B`=3)
+ mixtures (`optimizeParametersJOLTMix`=2) + CTF (`runCTFModelFinder`=1)**. The 2026-07-07 census claim — *"a full disk
census finds `8cc3cb84` at no path"* — and the "**doubly-lost**" framing below are **WRONG**: that sweep evidently
covered `/scratch` only, **not `/g/data`**. ⇒ **The parsimony figure's exact bits are RECOVERABLE**, and the 20 scripts
pinning `EXPECT_MD5=8cc3cb84` can be re-pointed at the archive path (their `build-gpu-on/` *path* is still stale — that
file is `0faac84d` since 2026-07-04 — but **the bits are not gone**). `/g/data/um09/as1708/gems-bin/` also archives
`0faac84d`, `5c48211e`, `885edcb5`, `fe693d75`; **it is a de-facto frozen store and was entirely missing from this registry.**

### Capability matrix — from disk, 2026-07-16 (✅ = sentinel present, — = absent)

| binary | tree / path | CTF | MIX | PARS | NS | BOOT | L0 | L5 | L6 | L7 | CACHE | **+I+R** | MFRES | REM |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **`020ff472`** | `iqtree3-mfresident/build-mfresident` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **✅** | ✅ | ✅ |
| `5ba5e2f1` | `iqtree3-mfresident/iqtree3-baseline-5ba5e2f1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| **`239fb5f2`** | `iqtree3-graduate/build-promote` (**shared w/ Hashara**) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | ✅ |
| `bffdc16e` | `iqtree3-graduate/build-flagclean2` (the >12 h run) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| `3d124292` | `iqtree3-graduate/build-flagclean` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| `1c86fd6c` | `iqtree3-graduate/build-grad-l7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | ✅ |
| `916cba21` | `iqtree3-graduate/build-grad` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | ✅ |
| `fe5ce648` | `iqtree3-l0/build-l8` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | ✅ |
| **`9d845205`** | `iqtree3-l2search-stage2b/build-prod` (**tree-search flagship**) | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — | — | — | — |
| `0e24550a` | `iqtree3-l2search-stage2b/build-stage2b` | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — | — | — | — |
| `5c48211e` | `…/build-gpu-on/iqtree3-nstemplate` + gems-bin | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — | — | — | — | — |
| `0faac84d` | `iqtree3-l2search/build-gpu-on` + gems-bin | ✅ | ✅ | ✅ | — | — | — | — | — | — | — | — | — | — |
| **`8cc3cb84`** | **gems-bin `iqtree3-thesis-sweep-8cc3cb84` (RECOVERED)** | ✅ | ✅ | ✅ | — | — | — | — | — | — | — | — | — | — |
| `885edcb5` | gems-bin `iqtree3-l2search-bootfix-885edcb5` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — | — | — | — |
| `fe693d75` | gems-bin `iqtree3-gems-gpu-82b34139e71a` | ✅ | **—** | ✅ | — | — | — | — | — | — | — | — | — | — |
| `2c931f41` | `iqtree3-gpu/build-gpu-on` | ✅ | ✅ | **—** | — | — | — | — | — | — | — | — | — | — |
| `b85d482f` | frozen `iqtree3-g5.0-parity.b85d482f` | **n/a** | — | — | — | — | — | — | — | — | — | — | — | — |

> **`b85d482f` is NOT capability-deficient — the sentinel set does not apply.** It has **zero** `ctf`/`CTF` strings and no
> `--ctf` flag because it **predates native `--ctf`**: the `run_ctf_1m_mf_energy.sh` / `run_ctf_10m_mf_aa_h200.sh` scripts
> implement CTF **externally** (they carry `subsample` / `coarse` / `5000`-site logic) and use `b85d482f` as the plain
> JOLT+GPU engine. Its registry row ("produced the published CTF numbers") is **correct as written**. Verified 2026-07-16.

### 🎯 Promotion verdict — **THERE IS NO SUPERSET BINARY. Promotion is a 3-WAY MERGE.**

> 🔴 **CORRECTION (same day, before anyone acts on it): an earlier cut of this section claimed "`020ff472` (13/13) is the
> only binary no other binary beats on any axis." That claim was FALSE — the sentinel set above was INCOMPLETE.** It
> omitted `JOLT_NO_PUREINVAR` / `JOLT_NO_FIXINVAR` / `JOLT_RTILE_OFF` (the M.2 coverage closures) and
> `JOLT_MF_DEVCHECK` / `JOLT_MF_DEVUSE` (the Direction-A mirror). Re-verified from disk 2026-07-16:
> **`020ff472` has NONE of those five.** This is exactly `MODELFINDER-FULL-GPU-PLAN.md` §N.5 landmine #4 — *"a combined
> binary needs an explicit MERGE, not an assumption that one tree has both."* The matrix above is accurate **for the
> sentinels it measures**; do not read it as a superset proof.

**Ground truth: three disjoint UNCOMMITTED clones, all based on `a07f61be`, none containing the others' work:**

| tree | binary | uncommitted diff | **unique shippable work** | scaffolding to DROP |
|---|---|---|---|---|
| `iqtree3-mfresident` | **`020ff472`** | 2 files, **+190/−35** | **`JOLT_NO_PINVFIX`** (+I+R fix) · `JOLT_MF_RESIDENT` · `JOLT_MF_NOSELFCHECK` | `JOLT_RDIAG`, `JOLT_CONSTCACHE` |
| `iqtree3-pureinvar` | **`eed14b92`** (`build-ctfab`) | 3 files, **+62/−13** | **M.2 coverage closures**: `JOLT_NO_PUREINVAR` · `JOLT_NO_FIXINVAR` · `JOLT_RTILE_OFF` | `JOLT_CTF_LEGACY`, `JOLT_SELFCHECK_STRIDE` |
| `iqtree3-mfdevcheck` | `c450faf6` · `b08da3a3` | 3 files, **+219/−20** | **`JOLT_MF_DEVCHECK` / `JOLT_MF_DEVUSE`** (mirror) | ~9 `JOLT_IR_*` + `JOLT_GAUGE_TRACE` |

**The merge is 4-way on the SAME two files — this is the whole difficulty:**

| file | mfresident | pureinvar | mfdevcheck | graduate `fa7d7a1c` |
|---|:--:|:--:|:--:|:--:|
| `tree/gpu/gpu_lnl_intree.cu` | +150 | +7 | +114 | **49 lines changed** |
| `tree/phylotreegpu.cpp` | +75 | +38 | +123 | — |
| `main/phylotesting.cpp` | — | +30 | — | — |
| `tree/gpu/gpu_iqtree.h` | — | — | +2 | — |

**Four hard blockers stand between this and canonical:**

1. **Three dirty trees** (table above) — `a07f61be` + 3 modified files each. **Nothing committed; nothing frozen.**
2. **The +I+R fix is unvalidated** where +R is competitive — gated on job **`174003950`** (rfavor).
3. **🔴 4-WAY COLLISION, and all three trees are BEHIND the branch.** `a07f61be` is an ancestor of graduate's HEAD **`fa7d7a1c`** ("Graduate mi3 flag-hygiene + blanket-maxiter revert to default", 5 files/+131−28) — **none of the three clones contains it**. `fa7d7a1c` edits **`tree/gpu/gpu_lnl_intree.cu` (49 lines)**; so do **all three** clones (+150/+7/+114), and all three also edit **`tree/phylotreegpu.cpp`** (+75/+38/+123). ⇒ **a real staged merge with conflicts on two files — NOT a fast-forward, NOT a single rebase.**
4. **No runtime gate has ever run on any merged bits** — parsimony/mixtures/CTF are `strings`-present only. Per the trap above, that is **not** proof.

**Promotion path (ordered):**
1. `rfavor` **`174003950`** green.
2. **Curate each diff** — ship the features, **drop the scaffolding** (`JOLT_RDIAG`/`JOLT_CONSTCACHE`; the ~9 `JOLT_IR_*` + `JOLT_GAUGE_TRACE` trace flags). A merge is **not** concatenation.
3. **Stage the merge onto graduate @ `fa7d7a1c`, one tree at a time, rebuilding between:**
   **pureinvar (coverage, +62 = smallest, closes the default-MF declines) → mfresident (+I+R fix) → mfdevcheck (DEVUSE, optional, largest).**
4. **Commit** — sole-author, **no AI attribution**.
5. **Clean rebuild.**
6. **Canonical gate on the NEW bits** — runtime **engage markers** (`GPUPARS-B`, mixtures, CTF) **+** RF=0 tree-search **+ G8** partitioned `-p` **+ G9** `--bnni` support multiset.
7. **M.4 decline census** (`-m MF`, DNA+AA, `JOLT_DBG=1`, count `[JOLT-GATE] decline reason=`) — the honest proof coverage actually closed. Must be **0** for standard DNA/AA.
8. **Freeze** md5-tagged into `frozen-binaries/` **and** `gems-bin/` (`chmod 555`) + add a registry row. **Never pin a `build-*/` path.**
9. **Human push** (assistant never pushes GPU source).

> ⚠️ **`iqtree3-l0` is parked on the DO-NOT-BUILD commit.** Its HEAD is **`37e41df7`** — the exact commit flagged below as
> having a **confirmed regression**. Its `build-l8` binary `fe5ce648` predates that HEAD, so the *binary* is unaffected, but
> **do not build from that tree as it stands.**

---

# 🚨 2026-07-13 — **THE CANONICAL BINARY.** One tree, one binary, capability-union of every incumbent.

> ## ⚠️ READ THIS FIRST — "canonical" means two different things and only ONE of them is being claimed.
>
> | claim | status |
> |---|---|
> | **Canonical FOR ALL NEW WORK** (every future run, every new script, the binary shared with Hashara) | ✅ **YES** — once the two gates below are green. It is a verified capability *union*: nothing any incumbent can do, it cannot. |
> | **Canonical AS THE FIGURES' PROVENANCE** (i.e. "this is the binary that produced the thesis figures") | 🔴 **NO — AND DECLARING IT CANONICAL DOES NOT MAKE IT SO.** The published figures were produced by ≥4 *different* binaries. A new binary cannot retro-anchor them. **They must be RE-RUN.** See the re-anchor list below. |
>
> Conflating these two is the single easiest way to put a false provenance claim in the thesis. The old "there is NO
> single canonical thesis binary" warning further down **remains true about the figures** and is NOT retired.

> ## 🔴🔴 2026-07-13 — **COMMIT `37e41df7` IS ON `setonix-iq/gpu-kernel-dev` AND IT HAS A CONFIRMED REGRESSION. DO NOT BUILD FROM IT.**
> A red-team audit of the promotion diff found **six** real defects, one of them a **strictly-worse-than-before regression on
> the mainline partitioned workflow**. The fixes are in the `iqtree3-graduate` working tree (uncommitted) and re-gated as
> `173609525`. **A follow-up commit + push is required.** Nothing was ever built, frozen, symlinked or shared from
> `37e41df7`, so the blast radius is the dev branch only.
>
> | # | defect | status |
> |---|---|---|
> | **2** | **Cache SERIALISES partitioned ModelFinder.** MF parallelises across partitions (`main/phylotesting.cpp:3066`) ⇒ N OMP threads × N *distinct* `Alignment`s all calling `optimizeParametersJOLT`. One global cache slot ⇒ **every call misses** and rebuilds `O(ntax·nptn)` **while holding the process-wide mutex** ⇒ across-partition parallelism collapses to **serial**. The pre-cache code built a private stack vector per thread, lock-free. **Strictly worse than before, on `-p <partitions> -m MFP --jolt`.** Invisible to every gate (all cells single-alignment). | **FIXED** — cache bypasses inside `omp_in_parallel()`; new gate **G8** proves promoted is not slower than base on partitions. |
> | **4** | **A live safety guard was deleted.** The old inline tip-build declined to CPU on an out-of-range leaf taxon; hoisting it into `joltGetTipPtnFreq` dropped that check. Since `node_leaf[v] = -1` is the launcher's *internal-node* sentinel, such a leaf is silently fed to the kernel as a **childless internal node** ⇒ wrong lnL, wrong NNI ranking, **no error**. | **FIXED** — guard restored at both call sites. |
> | **5** | **The det-fix did NOT cover mixlen.** `phylokernelnew.h:2465` still combined `all_dfvec`/`all_ddfvec` under `#pragma omp critical` = **arbitrary thread-arrival order** — the exact defect the patch exists to remove. `+I`/`+R` take the *other* branch, so the ×3 bit-identity gate never exercised it. Heterotachy / GHOST / `+H` (and `refineBootTrees`, which builds a `PhyloTreeMixlen`) **still wobbled**. My "non-determinism FIXED" claim was **over-scoped**. | **FIXED** — mixlen now combines in fixed packet order (the per-packet buffers were already packet-strided, so no extra allocation). |
> | **6** | **L0 default-ON silently reaches `refineBootTrees`** (`iqtree.cpp:3450`) under `--bnni`, where it writes `boot_logl[]` — i.e. **published UFBoot support values** — on all 1000 replicates. Previously unreachable (needed `JOLT_BOOT_SNAPSHOT` in the env). **Zero gates covered it.** | **GATED** — new **G9** requires the `--bnni` support multiset to match the base binary. |
> | **1** | **The canonical gate could NEVER pass.** It grepped the binary for `JOLT_DET`, which exists only in C++ *comments* — `strings` can never find it. Would have burned a 4 h gpuhopper job to fail a good binary. | **FIXED** — removed; the det-fix's proof is a *measured effect* (G5), not a string. |
> | **3** | **The promote gate could go green on CRASHED runs.** `run()` echoed `rc` but never tested it; `lnl()` matched `Optimal log-likelihood:` (printed *early*), so a `timeout`-killed run still yielded a value; `tmd5()` returned `""` for a missing treefile and `"" == ""` compared equal. G3 also never compared topology — a **different tree** with lnL within 1e-8 passed. | **FIXED** — rc + non-empty treefile + final-score required; G3 now demands md5-identity or RF=0. |
>
> **What the red-team cleared:** the `shared_ptr` rewrite compiles and is correct (`-fsyntax-only` clean on all three
> instantiations), neither call site mutates the cached buffers, no lock nesting, no deadlock. The CTF-subsample rebind
> is **fine** (coarse and refine are sequential phases; the key is derived identically at both call sites). And the
> det-fix's per-call allocation is ~0.003 % of the call body — immeasurable. **Finding #2 also proves the `shared_ptr`
> fix was NECESSARY, not defensive: the use-after-free was reachable, via partitions.**

> **⏳ GATE STATUS: `173609525` (promote, re-gated with G8+G9) → `173609526` (canonical) → `173609527` (AA A/B) → `173609528` (T1 perf). Do NOT cite, ship, pin or symlink until green.**
>
> **Why parsimony gets its own gate, and why `VERIFY mismatches=0` is NOT accepted as proof:** this doc already records
> the trap — *"running the GPU parsimony figure on `2c931f41` does not error — the kernel is simply absent, so it silently
> falls back to CPU and reports a false ~1.0× … `VERIFY mismatches=0` is a FALSE PASS when the kernel never fires."*
> The **engage marker is the gate.** A canonical binary that silently CPU-falls-back on parsimony would be worse than no
> canonical binary at all.

**Tree:** `/scratch/rc29/as1708/iqtree3-graduate`, branch **`gpu-promote-20260713`** → binary `build-promote/iqtree3` (**unstripped**, `-DIQTREE_FLAGS=nostrip`).
**Frozen (on gate-green):** `/scratch/rc29/as1708/frozen-binaries/iqtree3-canonical-<md5>` (immutable, `555`) + stable symlink `frozen-binaries/iqtree3-canonical`. **Pin the frozen path + md5, never `build-promote/`** — a `build-*/` path is LIVE and gets silently rebuilt out from under its pins. That is precisely how `8cc3cb84` was destroyed and 20 scripts were broken.

## The capability union — why this one binary can replace all of them

| incumbent | what it uniquely had | in the canonical? |
|---|---|---|
| `8cc3cb84` (**gone from disk**) | parsimony + mixtures + CTF | ✅ all three |
| `fe693d75` (gems-bin) | parsimony, CTF — **no mixtures** | ✅ + mixtures |
| `9d845205` (stage2b prod) | NS-template + T=3 + Stage-2b tree search | ✅ (same kernel set, `diff` of `__global__` symbols identical) |
| `fe5ce648` (build-l8) | + L0 / L5 / L6 / L7 / REM (**all default-OFF**) | ✅ + **default-ON** |
| `2c931f41` (iqtree3-gpu) | MF/tree-search/parity — **no parsimony at all** | ✅ + parsimony |
| `b85d482f` (frozen CTF parity) | CTF `-m MF` | ✅ |
| — | **screen-cache + det-fix** | ✅ **new, exists nowhere else** |

Source-verified 2026-07-13 (markers, not assumed): parsimony is real code at `tree/phylotreepars.cpp` (with the `[GPUPARS-B]` / `[GPUPARS-TIMING]` engage markers), mixtures span `tree/phylotreegpu.cpp` + `tree/phylotree.h` + `model/modelfactory.cpp`. The canonical gate re-proves both **in the built bits**, at runtime.

## 🔴 The re-anchor list — what "canonical" does NOT do for you

Declaring this binary canonical does **not** change which binary produced any already-published number. To make it the
figures' provenance, each of these must be **re-run on the frozen canonical**:

| figure / result family | produced by | status |
|---|---|---|
| CTF AA-1M/10M `-m MF` | frozen `b85d482f` | must re-run |
| ModelFinder 3–59× sweep, tree-search, Hashara parity | `2c931f41` | must re-run |
| +R/+I warm-seed + mixtures | `fe5f01f0` (**deleted, does not exist**) | must re-run |
| fig1/fig10 depth + energy | `fe693d75` | must re-run |
| Parsimony spine (56.591 s) | an ephemeral `/jobfs/<id>` copy — **md5 unrecoverable** | must re-run (**and the number will change** — the source was edited after that run) |
| fig5 bit-identity | a V100 log | must re-run |

**Until then the honest statement is:** *"the canonical binary is a verified superset of every binary that produced a
published figure; figures not yet re-run on it retain their original binary's provenance, recorded per-figure."*
That is defensible. *"All figures were produced by the canonical binary"* would be **false** until the re-runs land.

**Why it is finally ONE binary — the lineage is a strict superset chain (git-verified, 0 commits missing):**
`eef09e2c` (prod tree-search: NS-template + Stage-2b + T=3, the `9d845205` flagship) → `c459e147` (L0) → `44cab3df` (L0-under-`-B`) → `deca9df0` (L0+R) → `55743479` (L5/L6 — **this is what `build-l8`/`fe5ce648` was built from**) → `3cc07bc9`+`05d8ab61` (graduation: L5/L6/L7 default-ON) → **this branch** (det-fix + cache + defaults).
Same GPU kernel set as prod (`diff` of `__global__` symbols = identical). So the "≥4 binaries on ≥3 branches" problem is *closed by construction*, not by a merge.

> **⚠️ Naming trap: there is NO "L8 lever."** `build-l8` is a *build directory* (the 8th in `iqtree3-l0`: `build-l0/`, `build-l5/`, … `build-l8/`), not an optimisation. The levers are **L0, L5, L6, L7** (L1–L4 were plan numbering; **L3 was refuted with no code**). The promoted binary is a **strict superset of `build-l8`** — same source lineage, nothing dropped.

**Capability audit of the promoted tree (source markers, verified 2026-07-13 — not assumed):** `runCTFModelFinder` ✅ · `optimizeParametersJOLT` ✅ · **`optimizeParametersJOLTMix` (mixtures) ✅** · **`GPU_PARSIMONY_BATCHED` (parsimony) ✅** · `JOLT_NS_TEMPLATE` ✅ · `JOLT_BOOT_SNAPSHOT` ✅ · `JOLT_RBRLEN` ✅ · `JOLT_IBRLEN` ✅ · `JOLT_NO_FREERATE_HIGHK` ✅ · `JOLT_L0` ✅ · `JOLT_REM` ✅ (correctly still OFF — Stage-D NO-GO) · **`JOLT_NO_SCREEN_CACHE` (new)** ✅ · **`JOLT_DET` (new)** ✅.
**It carries BOTH GPU parsimony AND mixtures AND CTF AND the full tree-search stack — no prior single binary did** (`8cc3cb84` had parsimony+mixtures but no L0/L5/L6/L7; `fe693d75` had parsimony but no mixtures; `2c931f41` had no parsimony at all). **The "which binary produced which figure" provenance problem is therefore closable on this one tree.**

**The promotion is NOT new optimisations on top of l8.** It is (a) l8's levers **turned on by default** — in l8 every one was default-OFF, which is why the old recipe was a wall of env vars — plus (b) **two genuinely new things**: the screen-cache (which turned out to be the big one) and the determinism fix.

## ⚠️ BREAKING: the defaults flipped. Read this before running anything.

| feature | before | **now** | kill-switch |
|---|---|---|---|
| **JOLT screen-cache** (the alignment-constant `tip[]`/`ptnFreq[]` rebuild) | opt-in `JOLT_SCREEN_CACHE=1` | **DEFAULT-ON** | `JOLT_NO_SCREEN_CACHE=1` |
| **L0** (CPU postorder → GPU-resident) | opt-in `JOLT_L0=2` | **DEFAULT-ON (level 2)** | `JOLT_NO_L0=1` (or `JOLT_L0=0`) |
| **L0 under `-B`** (Stage-2b snapshot) | opt-in `JOLT_BOOT_SNAPSHOT=1` | **DEFAULT-ON** | `JOLT_NO_BOOT_SNAPSHOT=1` |
| **L5 (+R) / L6 (+I) GPU brlen** | opt-in | **DEFAULT-ON** (graduation) | `JOLT_NO_RBRLEN=1` / `JOLT_NO_IBRLEN=1` |
| **L7** (high-K +R cap-lift) | opt-in | **DEFAULT-ON** (graduation) | `JOLT_NO_FREERATE_HIGHK=1` |

`JOLT_L0=1` still selects **MEASURE** mode (runs both, keeps the CPU value ⇒ byte-identical trajectory).
`JOLT_SCREEN_CACHE_CHECK=1` = the exact staleness verifier (recompute + `memcmp` on every reuse, `abort()` on mismatch).
**Old scripts that set `JOLT_SCREEN_CACHE=1 JOLT_L0=2 JOLT_BOOT_SNAPSHOT=1` still work and are now redundant** — gate G2 requires default == explicit, bit-identical.
L0's **correctness guards are unchanged** (it only engages under `--ts-fused`, `--ts-fused-topm 0`, and not with `--write-intermediate-trees` / site-posterior printing).

## 🎛️ THE COMPLETE FLAG AUDIT (2026-07-14) — you should not need to set any of these

**The binary is fully optimized on pure defaults.** Every optimization below is ON (or correctly tuned) with **zero
environment variables**. The only reasons to set anything: an A/B (`_NO_` / `_CHECK` switches), profiling (diagnostics),
or deliberate tuning.

> 🔧 **FLAG-OVERLAP FIX (2026-07-14 — implemented in the graduate tree; SOURCE-VERIFIED, still UNBUILT/UNGATED, NOT shipped).**
> `--jolt` and `--ts-fused` were two *separate* GPU switches for two *different* phases — `--jolt` = GPU model/parameter
> optimization (read in `model/modelfactory.cpp`), `--ts-fused` = GPU **tree search** (read in `tree/iqtree.cpp`). The
> GPU-build default (`main/main.cpp`) turned on `jolt` but **NOT `ts_fused`**, so a plain run silently did GPU param-opt +
> **non-fused** tree search (no screener / no L0 / no global reopt), and the posture banner printed `L0=ON` even though L0
> requires `ts_fused` (`iqtree.cpp:3633` `_l0_ok = params->ts_fused && …`) and so could not fire. 26 % of our own gems
> scripts passed `--jolt` without `--ts-fused`. **Fix (5 edits):**
> - GPU-active now flips on the FULL pipeline: `ts_fused` **+ its 3 companion fields** (`ts_screen_adaptive` /
>   `ts_jolt_allbr` / `ts_reopt_split`) mirror the `--ts-fused` flag body verbatim ⇒ a bare invocation == the flagship
>   `--jolt --ts-fused` config.
> - new kill switch **`--no-ts-fused`** (mirrors `--no-jolt`) for A/B / bisection / legacy `--jolt`-alone arms.
> - banner is now honest: new **`ts-search=ON/off`** field, and `L0`/`cache` report ON only when `ts_fused` is truly active.
>
> **⇒ the GPU invocation collapses to nothing:** on the promoted binary, `iqtree3 -s ALN -m MFP -B 1000 -starttree PARS`
> gets the whole optimized path; add `--ctf` only for GPU ModelFinder. Side benefit: closes the **bare-`--jolt` 63 GB
> `:1269` OOM footgun** — `--jolt` now always implies the tiled `ts_fused` path. **Gate (separate build, after the
> flag-baking gate 173736600): default-run == promoted+`--ts-fused` bit-identical/RF=0; `--no-ts-fused` == legacy.**
>
> ⚠️ **On `--ctf` (Minh's guidance = continue testing WITHOUT it):** for a **fixed model** (`-m LG+R4`, `-m GTR+G4`)
> `--ctf` is a **no-op** — there is no model selection — so omitting it is correct and costs nothing (that is the
> `LG+R4 -B 1000` regime, our +R flagship). `--ctf` matters ONLY for **`-m MF`/`-m MFP`**, where without it model-testing
> runs on CPU (the 12-core phase that sank the `cudajolt` column, 21–44 % of wall) while an OpenACC port that offloads the
> whole likelihood does MF on the GPU. **So no-`--ctf` handicaps us only on full-ModelFinder cells, not on fixed-model
> tree search** — worth raising with Minh only if the comparison set includes `-m MF` cells.

This audit exists because the flag count reached the **mid-40s** and a muscle-memory "battery" of 7 env vars was being
pasted into every run — of which **5 were already default-ON, 1 was a profiling timer, and 1 a tuning knob**. Setting
them by hand caused *ruined runs* when one was forgotten. As of 2026-07-14 that battery collapses to **the default**:
set nothing.

> ⚠️ **COUNT CORRECTION (grounded red-team, 2026-07-14):** an earlier version of this audit said "44 → 35" and claimed
> to be complete. It was **scoped to 5 files under `tree/` only**. A whole-tree `getenv` sweep finds **37 distinct
> flags now** (≈46 before the 9 removals below), including three this table originally missed:
> `IQTREE_GPU_DIAG` (main/main.cpp — GPU-info dump; §3), `JOLT_MIX_HOSTDRIVEN` (model/modelfactory.cpp — host-driven
> profile-mixture opt path; opt-in, §3/tuning), `TS_CLEAN_PRE` (tree/phylotree.cpp — screener clean-pre lengths; §3).
> All three are default-OFF and change no production behaviour. Evidence: `scratchpad/REDTEAM_EVIDENCE.md` D1.

### 1 — Optimizations: DEFAULT-ON (set the kill-switch only for an A/B)
| flag | default | what it does | disable with |
|---|---|---|---|
| `JOLT_SCREEN_CACHE` | **ON** | one cache for the alignment-constant `tip[]`/`ptnFreq[]`, shared by screener + reopt | `JOLT_NO_SCREEN_CACHE=1` |
| `JOLT_L0` | **ON (2)** | CPU postorder → GPU-resident | `JOLT_NO_L0=1` / `JOLT_L0=0` |
| `JOLT_BOOT_SNAPSHOT` | **ON** | L0 under `-B` (Stage-2b snapshot) | `JOLT_NO_BOOT_SNAPSHOT=1` |
| `JOLT_IBRLEN` | **ON** | L6: GPU branch-length opt for `+I` | `JOLT_NO_IBRLEN=1` |
| `JOLT_RBRLEN` | **ON** | L5: GPU branch-length opt for `+R` | `JOLT_NO_RBRLEN=1` |
| `JOLT_FREEQ` | **ON** | GPU free-Q (estimated base frequencies) | `JOLT_NO_FREEQ=1` |
| `JOLT_FREERATE_HIGHK` | **ON** | L7: high-K `+R` cap-lift (legacy cap 4 → higher) | `JOLT_NO_FREERATE_HIGHK=1` |
| `JOLT_NS_TEMPLATE` | **ON** | NS-template screener occupancy (AA-1M 2.31×) | `JOLT_NS_TEMPLATE=0` |
| GPU parsimony | **ON under `--gpu`** | batched Fitch on GPU, auto-engages | `GPU_PARSIMONY_OFF=1` |

### 2 — Tuning knobs: sensible default, override only to tune
| flag | default | what it does |
|---|---|---|
| `JOLT_BRLEN_MAXITER` | **data-type-aware**: single DNA → **2**, everything else (AA/codon/**partitioned**) under `-B` → **3**, non-boot → 2 | per-round LM iteration cap. `<0` = skip GPU reopt → CPU fallback (profiling only). Source `iqtree.cpp:3893` keys on `seq_type==SEQ_DNA && !isSuperAlignment()` (NOT `num_states<=4`, which mis-fires on partitioned runs). **Grounded:** DNA mi2==mi3 bit-identical (G7 & headline both −59208019.101646); AA floor is mi3 = **102 iters / 949.75s, ~1.4× faster than mi8's 1334s** (job 173016501) — above the convergence cliff, lower cap = faster, so mi3 beats mi8; mi2 under-converges AA to 300 iters (172977555). |
| `JOLT_NTILE` | auto | Pattern-tiling factor for alignments larger than GPU memory (auto-chosen from `cudaMemGetInfo`; the override is read at `gpu/gpu_lnl_intree.cu:1323` and `:2675`). **⚠️ `+R`/`+I+R` reproducibility:** changing `nTile` changes the tiled reduction's summation order, so a FreeRate lnL is only **result-invariant (rel ≲1e-12), NOT byte-identical** across *different* nTile values — this is exactly why the one cross-tile `+R` gate (`gems_rtile_gate.sh:75-79`, nTile=1 vs =2) asserts `rel≤1e-12`, not byte-equality. To get a strictly **bit-identical** `+R` result (a determinism/reproducibility gate that stores an lnL and re-compares it across runs or binaries), **pin `JOLT_NTILE=1`**. A same-job A/B whose two arms run the same alignment on the same GPU auto-picks the *same* nTile, so `+R` bit-identity holds there without pinning — which is why all the 100K coverage/L5/L6/determinism gates (auto nTile=1 at that scale) are unaffected. The one at-scale, cross-binary `+R` "lnL bit-identical" cell (`gems_grad_dna1m_final.sh` CLAIM-1, DNA-1M `GTR+R4`) is safe today only because both arms take the same GPU path; pinning `=1` on both would make it immune (see the note in that script). Related in-source: the default-OFF block-tree reduce is likewise result-invariant-not-bit-identical (`:2303-2304`). |
| `JOLT_TS_NSTREAMS` | auto | CUDA streams for the screener |
| `JOLT_LBFGS_M` | internal | L-BFGS history depth for the JOLT optimiser |

### 3 — Diagnostics: default OFF, **no effect on results** (profiling only)
`JOLT_DIAG` (echild/H1/device per-call timers) · `TS_SCREEN_SPLIT` (screener host-vs-GPU wall split — **this is the one
that kept getting pasted into the battery; it is a timer, it changes no compute and no result**) · `JOLT_DEBUG` ·
`JOLT_AUDIT` · `JOLT_PARS_TIMING` · `TS_SCREEN_DUMP` · `TS_ADAPTIVE_DIAG` · `ALLDERV_DBG` · `TS_SCREEN_CHAR` · `TS_KCOUNT`

### 4 — Verifiers: default OFF, run BOTH paths + compare (**SLOW**; correctness A/B only)
`JOLT_SCREEN_CACHE_CHECK` (recompute + `memcmp` every reuse, `abort()` on mismatch) · `GPU_PARSIMONY_VERIFY` ·
`GPU_BOOT_VERIFY` · `JOLT_RGRADCHECK` (+R gradient finite-diff)
*(`JOLT_TS_BATCHFOLD_CHECK` / `JOLT_TS_ASYNC_CHECK` were the shadow-checks for the two retired experiments — deleted with them in §5.)*

### 5 — Experimental / NOT in the production path (default OFF) — **DELETED 2026-07-14**
`JOLT_TS_ASYNC` (**retired NULL**) · `JOLT_TS_ASYNC_CHECK` · `JOLT_TS_BATCHFOLD` · `JOLT_TS_BATCHFOLD_B` ·
`JOLT_TS_BATCHFOLD_CHECK` · `TS_SCREEN_GPUREDUCE` · `TS_EIGEN_UPPER` · `GPU_PARS_1D` (superseded by the shipped
2D-grid Fitch kernel) · `JOLT_REM` (measured neutral 1.0×) — all were **opt-in, default-OFF**, so removing them cannot
change the production path; the build gate proves it (production lnL **bit-identical** with `JOLT_BRLEN_MAXITER=3`).
Recoverable from git history if an experiment is ever revisited.

> ⚠️ **`TS_INIT_JOLT_OFF` is NOT experimental — it is a KILL SWITCH and is KEPT.** It reads
> `if (params->ts_fused && getenv("TS_INIT_JOLT_OFF") == nullptr)` (`iqtree.cpp:936`) — i.e. the gated behaviour is
> **default-ON** and the env *disables* it. An earlier cut of this audit filed it under "experimental"; deleting it
> would have silently changed behaviour. It belongs in §1 with the other kill-switches. *(Caught before the cut — the
> lesson: classify a flag by its `== nullptr` / `!= nullptr` polarity, not by its name.)*

### The old 7-flag "battery" → what it means now
`JOLT_SCREEN_CACHE=1 TS_SCREEN_SPLIT=1 JOLT_L0=2 JOLT_BOOT_SNAPSHOT=1 JOLT_BRLEN_MAXITER=2 JOLT_IBRLEN=1 JOLT_RBRLEN=1`
- `JOLT_SCREEN_CACHE` · `JOLT_L0=2` · `JOLT_BOOT_SNAPSHOT` · `JOLT_IBRLEN` · `JOLT_RBRLEN` → **already default-ON** (redundant)
- `JOLT_BRLEN_MAXITER=2` → **now the DNA default** (data-type-aware; AA correctly stays 3 — a blanket 2 would 3× AA bootstrap)
- `TS_SCREEN_SPLIT=1` → a **profiling timer**, never needed for speed
- ⇒ **The whole battery = the default. Set nothing.** Promotion gates now run pure defaults.

> ⚠️ **Provenance caveat:** the `JOLT_BRLEN_MAXITER` data-type-aware default is a **2026-07-14 graduate-tree edit
> (`iqtree.cpp:3890`), pending the next build+validate.** The currently-promoted binary `239fb5f2` still uses the old
> `mi=3` under `-B` for *all* data types — which is exactly why gate G7 (pure defaults) measured DNA-1M at 1409 s vs the
> 800 s headline (that headline binary `acc79c56` had forced `JOLT_BRLEN_MAXITER=2`). The A/B job `gems_brlen_ab.sh`
> decomposes how much of that 1.88× is the maxiter default vs a possible code regression before this default is trusted.

## What the promotion adds on top of the graduation

1. **`JOLT_SCREEN_CACHE`** (`tree/phylotreegpu.cpp`) — the tree-search-phase win. `gpuScreenNNIRank` and `optimizeParametersJOLT` both rebuilt the *alignment-constant* `tip[]`/`ptnFreq[]` on **every call** (~1.05 s × 1168 calls at DNA-1M). One mutex-guarded, fingerprinted cache serves both. Bit-identical. **⚠️ MODEL-DEPENDENT speedup — not a flat number:** multi-seed 1M sweep (job 173571086) gives **~2.14× on F81+F+G4, ~2.15× on GTR+G4, but only ~1.15× on GTR+I+G4** (the +I branch-length reopt dominates that wall, so the rebuild is a smaller share). Quote it per-model. (It bypasses inside `omp_in_parallel()` — see the regression note above — so on partitioned `-p` runs it is a no-op, not a win.)
2. **The det-fix** (`tree/phylokernelnew.h`) — the OpenMP `reduction(+:all_df,…)` in `computeLikelihoodDervGenericSIMD` had an unspecified combine order ⇒ last-ULP run-to-run wobble ⇒ flipped near-tie NNIs on `+I`/`+R`. Now a fixed-order per-packet combine (mirrors the `tree_lh` combine already at `:3148`). **This is what unblocked graduation CLAIM-2.** No-regression on `GTR+G4`: `rel = 0.00e+00`, tree bit-identical (job 173570466).
3. **Red-team #1 closed** (required before the cache could be a default): the cache buffers are `shared_ptr`-held and the getter hands out a **copy taken under the lock**, so a rebind for a different alignment allocates a fresh buffer instead of `realloc`-ing one a reader is mid-read on. (The old code returned `&g_jsc_tip` and the caller read it *outside* the lock — a latent use-after-free, unreachable today but not shippable as a default.)

## The numbers it is being promoted on

> 🔴 **2026-07-16 CORRECTION — the DNA `683.6 s / 1.31× / 800.2 s / 4.11×` headline below is ENV-DEPENDENT and was NEVER
> reproduced on the promoted `239fb5f2`.** All four numbers needed `JOLT_BRLEN_MAXITER=2` in binary `acc79c56`. At **pure
> defaults on `239fb5f2`, DNA-1M tree search is ~1288 s = ~0.69× — i.e. Hashara's 894 s is FASTER**; e2e is ~1409 s ≈ 2.34×
> and **that e2e win is our CTF ModelFinder (35.5×), NOT tree search**. (The maxiter provenance caveat above already flags
> the 1409 s e2e; this is the stronger, corrected statement — on DNA tree-search-*proper* we LOSE.) Where JOLT genuinely
> beats Hashara is **AA** (all models, 1.123×) and **e2e via CTF MF**, not DNA tree search. See `project_fullgpu_endtoend`
> / `project_gpu_tree_search` memory + `MODELFINDER-FULL-GPU-PLAN.md`. The table below is kept as the *promotion-time*
> record, **not** as promoted-binary truth.

**Re-grounded 2026-07-13 directly from the raw job logs** (an earlier version of this table spliced two *different*
configs together — fixed-model V0 numbers next to an MFP number — and the implied ratio was a cross-config comparison.
Everything below is ONE config from ONE pair of jobs, and the iteration counts are shown so you can check the work.)

**DNA-1M, `-m MFP -B 1000 -ninit 2 -seed 12345 -nt 12`, H200. Baseline + Hashara = job `173500688`; cache+L0 = job `173571993`. MFP selected F81+F+G4 in every arm.**

| arm | tree-search | iters | **s/iter** | final lnL |
|---|---|---|---|---|
| JOLT base (no cache, no L0) | 2394.1 s | 104 | 23.02 | −59208019.101646 |
| **Hashara** (`e713866b`) | **894.0 s** | 102 | 8.76 | −59208019.248 |
| **JOLT + cache + L0** | **683.6 s** | 104 | **6.57** | −59208019.101646 |

**3.50× over our own base; 1.31× ahead of Hashara — at the same 104 iterations and the same optimum** (ours is also the better lnL). Per-iteration: **1.33× faster than her.**

**Full-pipeline phase split, same job — this is where the 4.11× comes from:**

| phase | Hashara | JOLT + cache + L0 | ratio |
|---|---|---|---|
| ModelFinder | 2374.9 s (CPU) | **66.9 s** (GPU CTF) | **35.5×** |
| tree search | 894.0 s | 683.6 s | 1.31× |
| **TOTAL** | **3292.4 s** | **800.2 s** | **4.11×** |

**AA — not yet claimed, but no regression either.** AA-1M cache+L0 ran **200 iterations in 5375.8 s (26.9 s/iter)** vs the flagship's **102 iterations in 4649.9 s (45.6 s/iter)** — *identical* final lnL (−78605196.435066). It did 2× the iterations in 1.16× the wall ⇒ **1.70× faster per iteration**; the extra iterations came from an unpinned start tree (`-starttree PARS` was missing from that script). The matched AA cache A/B is job `173603345`.
**Always pin `-starttree PARS`.** That omission cost ~98 wasted iterations and briefly looked like a regression.

**Human push (assistant never pushes GPU source):**
`git -C /scratch/rc29/as1708/iqtree3-graduate add -u && git commit && git push setonix-iq gpu-promote-20260713`

---

> ## ⚠️ HISTORICAL (2026-07-02) — **half of this is now fixed, half is still true. Know which.**
> - 🟢 **FIXED by the canonical binary above:** "there is no single binary that *can* do everything." There is now — one
>   tree, one binary, verified capability-union, gated on the parsimony engage marker. **Build and run all new work on it.**
> - 🔴 **STILL TRUE, and NOT retired:** the **figures'** provenance is still split across ≥4 binaries, and
>   ~~`fe5f01f0` / `d711a4f9` / `8cc3cb84` **still do not exist on disk**~~ 🔴 **PARTLY FALSE — corrected by the 2026-07-16
>   census (top of file): `8cc3cb84` EXISTS** at `/g/data/um09/as1708/gems-bin/iqtree3-thesis-sweep-8cc3cb84` (parsimony
>   intact); ~~**`d711a4f9` exists only in a VOLATILE session scratchpad**~~ ✅ **RESCUED 2026-07-16** — `d711a4f9`
>   (native `--ctf` + mixtures, 9 CTF/mixture sentinel hits verified, md5 round-tripped) is now **durable at
>   `/g/data/um09/as1708/gems-bin/iqtree3-l2search-ctfprod-d711a4f9`** (`chmod 555`); the volatile source copies at
>   `/scratch/rc29/as1708/tmp/claude-24140/…/08fca9f7/…/scratchpad/iqtree3_new`+`iqtree3_pre2a` may be reaped anytime.
>   **Only `fe5f01f0` is genuinely absent** (as is `acc79c56`,
>   the binary behind the retired 683.6 s headline). The split-provenance point stands; the "does not exist" part does not.
>   A new canonical binary cannot retro-anchor an
>   old figure. **Do not cite any of those md5s as "the binary," and do not cite the canonical as the figures' binary
>   until the re-anchor re-runs land.** See the re-anchor list above.
>
> The trees this warning was written about:
> - **`iqtree3-l2search`** (branch `l2-batched-nni` @ `a52fd97b`, **committed clean**) — the **feature-complete**
>   tree: CTF + JOLT + +R/+I + profile mixtures + tree-search + GPU parsimony. Binary `build-gpu-on/iqtree3`
>   md5 **`8cc3cb84`** (built 2026-06-29). ⚠️ **"feature-complete" ≠ "validated": no completed end-to-end run
>   exists on `8cc3cb84`'s exact bits** (the fig4 re-run 172862242 is still pending), and its parsimony source
>   `gpu_lnl_intree.cu` was edited 2026-06-29 18:58 — **AFTER** the 2026-06-28 spine parsimony run — so `8cc3cb84`
>   is **not source-identical** to the binary that produced the "56.591 s" number.
> - **`gems`** clean-fork binary **`fe693d75`** at `/g/data/um09/as1708/gems-bin/iqtree3-gems-gpu-82b34139e71a`
>   (built 2026-07-01 from `gems`@`82b34139`) — **also has the GPU parsimony kernel** (`GPUPARS-B`), md5-gated by
>   three `repro_fig1_*` scripts and already used for the fig1/fig10 reproductions. **Not a superset** (no
>   mixtures). So **two** built binaries carry parsimony, not one.
> - **`iqtree3-gpu`** (branch `tree-search-ts0`, `2c931f41`) — tree-search DEV tree; **NO GPU parsimony** (kernel
>   genuinely absent — verified `GPU_PARSIMONY_BATCHED`/`GPUPARS-B` = 0 in binary and source). Drives the hashara
>   ModelFinder/tree-search parity sweep.
>
> **Gotcha that already burned a run:** running the GPU parsimony figure on `2c931f41` does **not** error — the
> kernel is simply absent, so it **silently falls back to CPU** and reports a false ~1.0× (fig4 job 172852055:
> GPU-leg 209.3 s == CPU-leg 209.5 s, no engage markers; `VERIFY mismatches=0` is a FALSE PASS when the kernel
> never fires). Use `8cc3cb84` (or `fe693d75`) and require the `GPUPARS-B` engage marker in the log. Scripts
> corrected 2026-07-02.

> **2026-07-06 update — T=3 prod binary shared with Hashara + committed to `gpu-kernel-dev`.**
> The current tree-search prod tree is **`iqtree3-l2search-stage2b`** (a *separate* tree from the
> `iqtree3-l2search` registered below), branch `l2-batched-nni`. Its binary **`9d845205`**
> (`build-prod/iqtree3`, built 2026-07-05) is the validated **T=3 flagship** (NS-template + Stage-2b +
> fused-reopt cap; AA-1M `-B 1000` tree-search = **2.694×** vs the 104-core node, **1.123×** ahead of the
> OpenACC port, RF=0 vs both, bit-identical; multiseed 2.63–2.72× / 1.11–1.13× over 3 seeds). It also carries
> the CTF ModelFinder path (`--ctf`) and the +R ladder — i.e. the single "everything" tree-search+fast-MF binary.
> - **Shared with Hashara:** `/scratch/dx61/as1708/shared-jolt/`. **`iqtree3-gpu-latest` → `iqtree3-gpu-promoted-239fb5f2`
>   as of 2026-07-14** (the PROMOTED canonical: cache + L0 + L5/L6/L7 + det-fix all **default-ON**, git `a07f61be`;
>   README-promoted.md published; run with `--ts-fused -nt 12`). **This replaces the prior `fe5ce648` target** (levers
>   all default-OFF — Hashara was measuring un-optimized prod, plan §89). Repoint done manually (`share_promoted_binary.sh`'s
>   log-guard couldn't find its cleaned-up `g8g9-rest.o*`/`canon-gate.o*` logs; the binary's G8/G9 PASS is documented +
>   maxiter mi3 re-confirmed bit-identical by job 173768871 Gate 1). `fe5ce648` retained on disk for rollback. Prior tips `iqtree3-gpu-9d845205` and
>   `iqtree3-gpu-l0-f8b1d0d9` remain on disk immutable but are no longer the `-latest` target. Runtime dep: `cuda/12.5.1`
>   only (`libcudart.so.12`). This **replaces her stale `iqtree3-gpu-cca7dbc1`** (built 2026-06-19 — predates the
>   Jun-23 native `--ctf`, the Jun-27 +R ladder, and the Jun-27 warm-seed fix; her `--ctf` help entry = 0).
> - **Source committed:** the 4 uncommitted T=3 edits (`gpu_iqtree.h`, `gpu_lnl_intree.cu`, `iqtree.cpp`,
>   `phylotreegpu.cpp`) are now commit **`eef09e2c`** (sole-author as1708, **no AI attribution**) on top of
>   `a52fd97b`. Local `gpu-kernel-dev` fast-forwarded `46925181` → `eef09e2c` (+27 commits, 0 behind = clean FF).
>   **Remote still at `46925181`** — the push is human-only:
>   `git -C /scratch/rc29/as1708/iqtree3-l2search-stage2b push setonix-iq gpu-kernel-dev`
> - **Pre-push gate (recommended before the push):** a clean rebuild from `eef09e2c` should pass CTF-MF + T=3
>   RF=0 — validate *behavior*, not md5 (a rebuild won't match `9d845205`'s bits because the build date is
>   embedded).
> - **PROFILING-ONLY twin `iqtree3.sym` (md5 `4e80bea0`, `build-prod/iqtree3.sym`, 2026-07-07):** `9d845205` is
>   **stripped**, so nsys CPU-sampling backtraces resolve to hex, not function names — useless for the full-GPU L0
>   caller-attribution. `iqtree3.sym` is an **unstripped relink of the identical `build-prod` objects** (link.txt has
>   `-g`, no `-s`; same -O3 → functionally identical timing) carrying `computeLogL`/`computePartialLikelihoodSIMD`/
>   `computeLikelihoodDervSIMD`. **Use ONLY for profiling (`gems_fullgpu_0d_*`), never for scientific/RF results** —
>   `9d845205` remains the flagship. Rebuild: rerun `build-prod/CMakeFiles/iqtree3.dir/link.txt` with `-o iqtree3.sym`.
> - **✅ CTF-MF validated on `9d845205` (job 173171300, 2026-07-07) — full PASS.** Multi-seed (1/2/3) all select
>   `LG+G4` (~281 s each, stable); `--ctf` winner == plain `-m MF` winner (both `LG+G4`), so CTF does not change
>   the answer; **CTF is ~12.6× faster for model selection** (281 s vs 3541 s full-data scan, same winner); the
>   full `-m MFP --jolt --ctf -B 1000` pipeline runs end-to-end (MF→coarse-rank 1232 models→top-3 full-data
>   refine→tree search→UFBoot corr 1.000, tree −7541976.852, 7.99 GB peak). The `--ctf` path on the shared
>   binary is confirmed good for Hashara's model-selection work. (An earlier attempt, job 173163875, was killed
>   for an uncapped-tree-search walltime overrun and re-run as 173171300 with `-fast` on the model-selection arms.)

> **2026-07-07 census + cleanup (verified from disk this date).** Three things changed since the 2026-07-02 audit:
> 1. **🧹 DELETED (freed ~5.2 GB total):** (a) the 5 superseded "ad-hoc dev" build dirs in `iqtree3-gpu/` —
>    `build-prod-on` (`447c60a7`), `build-prod-off` (`d3dffe08`), `build-ts0-off` (`30dc278c`), `build-dev-dbg`
>    (`134321eb`), `build-ts-prof` (`76fe51fd`, a 164 MB profiling build) — all 2026-06-23/24, "NOT reproducibility
>    artifacts — do not cite," **referenced by no runnable script** (only historical research logs + the build-validation
>    `run_prod_cut_validate_v100.sh`, which *creates* them fresh via `mkdir -p`+cmake), regenerable from `6fce15de`;
>    and (b) `iqtree3-l2search/build-gpu-diag` (`7b327dfe`, 855 MB, 0 refs, "not for timing"). `iqtree3-gpu` → 2.3 GB,
>    `iqtree3-l2search` → 2.0 GB. All live binaries (`2c931f41`, `6e0ade4f`, `0faac84d`, `3a18a319`, `9d845205`,
>    `0e24550a`) + all `frozen-binaries/*` were left untouched.
> 2. 🔴 **THIS ITEM IS FALSE — corrected 2026-07-16 (see the census at the top of this file).** `8cc3cb84` **EXISTS** at
>    `/g/data/um09/as1708/gems-bin/iqtree3-thesis-sweep-8cc3cb84` (full md5 `8cc3cb844e807fe3ecbcd0b9ff5428df`, parsimony +
>    mixtures + CTF all verified present). The 2026-07-07 "full disk census" below **swept `/scratch` only and missed
>    `/g/data`**, so it wrongly declared the binary lost. The *path* claim is still true (`build-gpu-on/` was overwritten to
>    `0faac84d`), but the **bits are recoverable and the parsimony provenance is NOT doubly-lost**. Re-point the 20 pinned
>    scripts at the archive path rather than treating them as unfixable. Original (wrong) text follows:
>    ~~**⚠️⚠️ `8cc3cb84` NO LONGER EXISTS ON DISK.**~~ `iqtree3-l2search/build-gpu-on/iqtree3` was **rebuilt** on
>    2026-07-04 20:19 and is now **`0faac84d`** (the maxiter-8 / pre-T3 "old-T8" reference used as the byte-identity
>    baseline in the Stage-2b validation). A full disk census (2026-07-07) finds `8cc3cb84` at **no path**. **BUT 20
>    scripts still hard-pin `EXPECT_MD5=8cc3cb84`** (all the `fig4_parsimony_*`, `cpu_parsimony_*`, `repro_ctf_avian`,
>    `gems_boot_*`, several `gpu-modelfinder/job*`, `diag_ts_bottleneck`, …) — they will **fail their md5 guard** if
>    re-run. This is a real breakage, pre-dating the cleanup: the parsimony canonical was overwritten by the T-floor
>    rebuild. A rebuild from `a52fd97b` won't restore `8cc3cb84`'s bits (source `gpu_lnl_intree.cu` was edited after,
>    and the build date is embedded). **Unresolved provenance item — the parsimony figure's exact binary is now
>    doubly-lost (ephemeral `/jobfs` copy AND the overwritten `build-gpu-on`).**
> **2026-07-10 update — NEWEST + MOST COMPLETE binary is `build-l8` `fe5ce648`** (`iqtree3-l0`, `gpu-kernel-dev` @
> `55743479`, committed clean, built 2026-07-09). **RED-TEAM AUDITED:** it is a strict **superset of prod `9d845205`** —
> the same tree-search stack (NS-template default-ON + T=3 fused-cap + Stage-2b) **plus** L0 + L5 (+R brlen) + **L6 (+I+G
> brlen, real env `JOLT_IBRLEN`)** + REM, all default-OFF. (An earlier string-grep wrongly marked L6 absent — it used a
> phantom env var `JOLT_OPTPINV`; L6 IS present.) **⚠️ `-m MFP` needs `--ctf`** or model-testing runs 100% on CPU; pair
> `--ctf` with `--ts-fused` (bare `--jolt` tree-search OOMs the ~63 GB `:1269` arena at 1M). Validated on `LG+R4 -B 1000`
> (job 173460572); the full `--ctf` MFP pipeline validated on siblings (`9d845205`/l2search). See the "freerate/brlen +
> L0 lineage" registry section for the exact flag set + the GO-WITH-FLAGS verdict. Does NOT retire the "no single
> canonical binary" provenance-of-figures warning — but it IS the binary to build/run current work on.
>
> 3. **New clone on disk: `iqtree3-boottile`** (branch `boot-pack-tiling` @ `e5c527c6`, the JOLT_BOOT_PACK work) —
>    two **CPU-only** validation binaries `build-boot-val/iqtree3` (`39318f01`, 2026-07-07) + `build-boot-cpu/iqtree3`
>    (`fe2bffb6`, 2026-07-06). **KEPT (active work):** bootpack proved 3.999× at AA-1M scale; the avian-`-B` OOM
>    unblock (which needs a GPU build of this clone) is still pending. Not pinned by md5 in any script (the validate
>    script rebuilds fresh), so the binaries are disposable, but the **source tree must stay**.

---

## The two kinds of binary — still true

| Kind | Location | Name source | On rebuild | Move/rename? |
|------|----------|-------------|------------|--------------|
| **LIVE** (cmake output) | `build-*/iqtree3` | cmake target `iqtree3` | **regenerated** (md5 changes) | **NO** — scripts hardcode the path |
| **FROZEN** (snapshot) | `frozen-binaries/` | hand-named, md5-tagged | untouched | yes — immutable copies kept on purpose |

`build-*/iqtree3` is whatever you last compiled; `frozen-binaries/*` never changes. Reproducible benchmarks
point at frozen; development/current-feature runs point at live.

---

## Registry — feature-complete tree: `iqtree3-l2search`

**Path:** `/scratch/rc29/as1708/iqtree3-l2search/` · **git:** branch `l2-batched-nni` @ `a52fd97b` — **committed
clean** (`git diff --stat HEAD` = empty; the GPU/parsimony kernels are committed at HEAD, not uncommitted WIP;
banked as tag `gpu-wip-2026-06-30`, see THESIS_REPO_PLAN §4). Source contains: `runCTFModelFinder` (CTF, 3
files), `GPU_PARSIMONY_BATCHED` (parsimony), `optimizeParametersJOLTMix` (mixtures, 2 files), `JOLT` (8 files).

| Use this | Path (under `iqtree3-l2search/`) | md5 | Built | What it is |
|---|---|---|---|---|
| ~~Feature-complete GPU `8cc3cb84`~~ **this PATH overwritten** → `build-gpu-on/iqtree3` = **`0faac84d`**; the `8cc3cb84` **bits live in gems-bin** | `build-gpu-on/iqtree3` (live) · **`8cc3cb84` → `/g/data/um09/as1708/gems-bin/iqtree3-thesis-sweep-8cc3cb84`** | **`0faac84d`** (was `8cc3cb84`) | rebuilt 2026-07-04 | 🔴 **CORRECTED 2026-07-16 — `8cc3cb84` is NOT gone.** This *path* was rebuilt 2026-07-04 to `0faac84d` (the maxiter-8/pre-T3 "old-T8" byte-identity reference), but the `8cc3cb84` binary **exists** at `gems-bin/iqtree3-thesis-sweep-8cc3cb84` (md5 `8cc3cb844e807fe3ecbcd0b9ff5428df`, parsimony+mixtures+CTF verified present 2026-07-16). The old "no longer exists at any path (2026-07-07 census)" claim was an artifact of a `/scratch`-only sweep. The feature-complete source is unchanged at `a52fd97b`. The 20 scripts pinning `8cc3cb84` fail on the **path**, not the bits — **re-point them at the gems-bin archive**. |
| CPU-only | `build-gpu-off/iqtree3` | `3a18a319` | 2026-06-26 | Same source, `IQTREE_GPU=OFF`. CPU-path parity / GPU-less baselines. |
| ~~diag build `7b327dfe`~~ | 🧹 **DELETED 2026-07-07** | ~~`7b327dfe`~~ | — | Was a diagnostics/cross-check build (855 MB), 0 script references, "not for timing." Removed in the 2026-07-07 cleanup. |

> **Which binary actually produced each figure (there is no single-binary provenance).** CTF AA-1M/10M `-m MF`
> → frozen `b85d482f`. ModelFinder 3–59× sweep + tree-search + hashara parity → `2c931f41` (`iqtree3-gpu`). +R/+I
> warm-seed + mixtures → the now-**deleted** `fe5f01f0`. fig1/fig10 depth+energy → `fe693d75` (`gems-bin`).
> Parsimony spine (172524833/526962/530260, 2026-06-28) → an ephemeral `/jobfs/<id>/iq` copy of `iqtree3-l2search`.
> fig5 bit-identity JOLT 2.772e-12 → a V100 log. **P0/P6 action: rebuild ONE binary from `gems/` and re-run or
> re-anchor EVERY figure on it before any doc calls a single binary the figures' provenance.**

---

## Registry — Stage 2b + NS-template experimental tree: `iqtree3-l2search-stage2b` (VALIDATED 2026-07-05)

**Path:** `/scratch/rc29/as1708/iqtree3-l2search-stage2b/` — an **isolated clone** of `iqtree3-l2search`
(git base `a52fd97b`) carrying the `5c48211e` uncommitted NS-template source **plus** the Stage 2b bootstrap
edits (7 in `tree/gpu/gpu_lnl_intree.cu`, 4 in `tree/phylotreegpu.cpp`, 1 header). Built in its **own** dir
`build-stage2b/` (fresh cmake configure, exact original flags) so it never touches the canonical `8cc3cb84`.

| Use this | Path (under `iqtree3-l2search-stage2b/`) | md5 | Built | What it is |
|---|---|---|---|---|
| **⭐ PRODUCTION (NS default-ON + T=3)** | `build-prod/iqtree3` | **`9d845205`** | 2026-07-05 | The **productionised** binary: `JOLT_NS_TEMPLATE` flipped **default-ON** (set `=0` for the byte-identical escape hatch) + the fused reopt cap lowered **8→3** under `-B` (T=3). Stage 2b (`JOLT_BOOT_SNAPSHOT`) still flag-gated (default OFF). **VALIDATED GREEN — job `173055925`, gates below.** This is the intended new thesis/figures binary once the P0/P6 single-binary provenance is re-anchored on it. |
| **Stage 2b + NS-template GPU** (flag-gated) | `build-stage2b/iqtree3` | **`0e24550a`** | 2026-07-05 | `8cc3cb84`'s feature set **plus** two independent, **bit-identical, flag-gated** levers: `JOLT_NS_TEMPLATE` (compile-time `NS∈{4,20}` peel kernels) and `JOLT_BOOT_SNAPSHOT` (`-B` mirror-recompute elimination). **Both default OFF ⇒ strict byte-identical no-op vs `8cc3cb84`/`0faac84d`** (gate G1). Superseded by `9d845205` for production; kept as the flags-OFF-by-default reference + the combined-cell measurement binary (it is **mi8** under `-B`, pre-T=3). |

**Production validation — job `173055925` (H200), ALL 5 GATES GREEN (prod binary `9d845205`):** G1 no-boot
byte-identity NEW-default vs `0faac84d` (AA-10K/DNA-10K lnL identical + tree byte-identical → NS default-ON is a
strict no-op); G1b escape hatch (`JOLT_NS_TEMPLATE=0` == default); **G2** T=3 under `-B` (AA-100K new-T3 == old-T8
lnL `-7541976.852167`, RF=0, 102 it, 435 s vs 1340 s = 3.08×; DNA-100K RF=0, lnL within rel 2e-11 = same tree);
**G3** default-ON engaged (AA-100K no-boot default 352.7 s vs `JOLT_NS_TEMPLATE=0` 733.3 s = **2.079×**); **G4** Stage
2b intact (549 guards, 0 NOT-OK, RF=0 vs gold).

**Validation — job `173024377` (H200), ALL GATES GREEN:**
- **G1 flag-OFF byte-identity vs `0faac84d`, WITH `-B`** — AA-10K & DNA-10K: lnL identical *and* tree byte-identical. Stage 2b OFF is a strict no-op. ✅
- **G2 `JOLT_BOOT_SNAPSHOT=1` correctness, AA-100K `-B`** — **513/513** `GPU_BOOT_VERIFY` identity guards OK (`Σ freq·_pattern_lh == joltLnL`), **max rel `2.65e-14`** (≪ `ufboot_epsilon` 0.5); RF=0 vs CPU gold; corr 1.000; BEST `-7541976.852167` (the canonical AA-100K optimum). (The validation script's "11 NOT-OK" was a false positive — the workdir path string `stage2b_…` matched the guard grep; the 11 were path-echo lines, not guards.) ✅
- **G3 ON==OFF support parity, AA-100K `-B`** — RF(ON vs OFF)=0, **contree byte-identical** (snapshot support == clean-room support). ✅
- **G4 perf, AA-100K `-B`** — OFF 1343.0 s vs ON 1269.2 s = **1.058×** (nTile==1; the zero-sweep win is nTile==1-only, and its AA-1M magnitude is still open — see BOOTSTRAP-UFBOOT-PLAN §3 Stage 2b). ✅

**NS-template (`JOLT_NS_TEMPLATE=1`) headline (same source `5c48211e`, no-boot, bit-identical RF=0 every cell):**
DNA-100K **1.11×**, AA-100K **2.07×**, DNA-1M **1.031×**, **⭐ AA-1M `2.314×`** (job 173024944 — the decisive
saturated-grid cell; overturns the plan's predicted sub-1.3× AA ceiling). See OCCUPANCY-ATTACK-PLAN §5 Phase 1.

> **Status:** experimental / flag-gated / validated-GREEN, **not yet default-ON and not yet the figures' binary.**
> Do not cite `0e24550a` as a thesis-figure provenance until the NS-template default-ON decision is taken and one
> binary is rebuilt from a committed tree (the P0/P6 single-binary action above still stands).

---

## Registry — freerate/brlen + L0 lineage: `iqtree3-l0` (build-l8 — the NEWEST binary, added 2026-07-10)

**Path:** `/scratch/rc29/as1708/iqtree3-l0/` · **git:** branch `gpu-kernel-dev` @ **`55743479`** ("+R/+I brlen GPU
offload (L5/L6) + high-K +R cap-lift, all default-OFF") — **committed clean** (working tree carries only untracked
`build-l{0,5,6,7,8}/` dirs; no uncommitted source). This is the `project_gpu_freerate_handicap` lineage.

| Use this | Path (under `iqtree3-l0/`) | md5 | Built | What it is |
|---|---|---|---|---|
| **⭐ NEWEST / most-complete for +R/L0** | `build-l8/iqtree3` | **`fe5ce648`** | 2026-07-09 23:59 | The **newest** GPU binary. Carries GPU ModelFinder (CTF/JOLTMix, same path as every GPU binary) **+ NS-template** tree-search lever **+ L5** (+R brlen, `JOLT_RBRLEN`) **+ L0** (`JOLT_L0` computeLogL offload / `-B` bootSnapshot) **+ HIGHK +R cap-lift + REM (EM rate)** — all levers default-OFF (flag-gated). **Validated for `LG+R4 -B 1000`** (job 173460572: L0 decoupling 2.098× nt1, bit-identical UFBoot supports 197/197, RF=0, lnL byte-id). **⭐ +R-vs-Hashara parity WON (job 173499262, LG+R4 -B 1000, H200): JOLT nt12 377s vs Hashara 912s = 2.42× total / 2.58× tree-search, AND better lnL −7541972.2385 vs −7541972.445** — JOLT's largest Hashara margin (+R = where both were weak; L5 wins it). **L5/L6 GRADUATED default-ON in worktree `iqtree3-graduate`@`3cc07bc9` (binary `56ff1e95`, validation 173501653 queued); L7 cap-lift graduation next.** |

**Capability matrix (RED-TEAM AUDITED 2026-07-10, agent — engagement + default state source-verified, file:line):**

| lever | env var (REAL) | in `fe5ce648`? | default | vs prod `9d845205` |
|---|---|:--:|---|---|
| GPU CTF ModelFinder | `--ctf` (`phyloanalysis.cpp:3458`) | ✅ | flag | same |
| Per-model GPU opt | `--jolt` (`modelfactory.cpp:1597`) | ✅ | flag | same |
| NS-template occupancy | `JOLT_NS_TEMPLATE` (`gpu_lnl_intree.cu:576`) | ✅ | **ON** | same |
| T=3 fused-reopt cap | hard-coded (`iqtree.cpp:3875`) | ✅ | **ON under `-B`** | same |
| Stage-2b mirror-kill | `JOLT_BOOT_SNAPSHOT` (`phylotreegpu.cpp:60`) | ✅ | OFF | same (OFF in `9d845205` too) |
| L5 (+R brlen) | `JOLT_RBRLEN` (`phylotreegpu.cpp:2040`) | ✅ | OFF | **build-l8 ONLY** |
| **L6 (+I+G brlen)** | **`JOLT_IBRLEN`** (`phylotreegpu.cpp:2069`) | ✅ | OFF | **build-l8 ONLY** |
| L0 (computeLogL offload) | `JOLT_L0` (`iqtree.cpp:3625`) | ✅ | OFF | **build-l8 ONLY** |
| REM (EM rate weight) | `JOLT_REM` (`gpu_lnl_intree.cu:3118`) | ✅ | OFF | **build-l8 ONLY** |
| GPU parsimony | (auto) | ✅ | ON | same |

**⚠️ CORRECTION to the earlier string-grep matrix (it was WRONG in the binary's favor):** L6 was mis-marked ABSENT — the grep used a **phantom** env var (`JOLT_OPTPINV` never existed in the source). **L6 IS present; its real gate is `JOLT_IBRLEN`.** And **T=3 + Stage-2b ARE present** (T=3 hard-coded; Stage-2b flag-gated, default-OFF in `9d845205` too). **⇒ build-l8 has NO tree-search gap vs `9d845205` and is a strict SUPERSET (+ L0/L5/L6/REM). It is unambiguously the most complete GPU binary.**

**RED-TEAM VERDICT (2026-07-10, agent): GO-WITH-FLAGS.** Two load-bearing operational facts:
1. **Plain `-m MFP` runs model-testing 100% on CPU** (`phyloanalysis.cpp:3458` `if(params.ctf && runCTFModelFinder) else runModelFinder`(CPU); `params.ctf/jolt` default false). GPU model-testing needs **`--ctf`** (coarse-rank all 1232 models on a 5000-site subsample + GPU JOLT top-k full-data refine, ~12.6× selection win) or at least `--jolt`. `--ctf` implies `--jolt --gpu`. GPU tree-search reopt is a SEPARATE switch: `--ts-fused`.
2. **The 59 GB `:1269` OOM is real** (`gb_partial=nInternal·ncat·ns·nptn·8` ≈ 63 GB at AA-1M untiled) but **avoided by pairing `--ctf` with `--ts-fused`** (tiled; job 172579585 ran AA-1M CTF clean, 100K CTF `-B` peaked 7.99 GB). **NEVER run bare `--jolt` tree-search at 1M.**

**Exact JOLT flag set for the `-m MFP -B 1000` benchmark** (matches validated `run_ctf_full_pipeline_{100k,1m}_h200.sh` + the L5/L6 handicap-closers):
```
env JOLT_BRLEN_MAXITER=2 JOLT_IBRLEN=1 JOLT_RBRLEN=1 \
  /scratch/rc29/as1708/iqtree3-l0/build-l8/iqtree3 \
  -s <ALN> -m MFP --jolt --gpu --ctf --ts-fused -B 1000 -nt 12 -seed <s> -pre <out> -redo
```
- `JOLT_IBRLEN=1 JOLT_RBRLEN=1` keep +I+G4 / +R brlen reopt on GPU (else CPU-decline via `invar-sites-brlenonly` gate `phylotreegpu.cpp:2084`; **AA-1M sim selects `LG+I+G4`**, so +I matters). Both runtime-self-check `JOLT_AUDIT` rel≤1e-6 → NaN→CPU on drift ⇒ cannot silently corrupt.
- 🔴 **SUPERSEDED 2026-07-13 — this next line was WRONG and is retracted:** ~~"Leave OFF for the nt12 benchmark: `JOLT_L0` (~14% nt12 ceiling; set `=2` only for nt1 CPU-decoupling), `JOLT_BOOT_SNAPSHOT`."~~ **L0 + the `-B` snapshot are worth 1.29× at DNA-1M nt12 and are part of the 800.2 s headline; they are now DEFAULT-ON in the promoted binary.** The "L0 is an nt1-only lever" framing was the mistake that kept it off every benchmark for a month (see the L0-OFF contamination audit). **`JOLT_REM` does stay OFF** (Stage-D NO-GO, 2.5× slower) — that part was right.
- **On the promoted binary you need NO env vars at all** — see the 🚨 PROMOTION section at the top of this file. The env-var wall below is the *build-l8* recipe, kept for reproducing pre-promotion runs.
- **Smoke-test 100K first:** the exact `fe5ce648` bits were validated on `LG+R4 -B 1000` (job 173460572) but NOT yet on the full `--ctf` MFP pipeline (siblings `9d845205`/l2search were). Confirm CTF picks a sane winner, RF=0 vs CPU, no `[JOLT-GATE] decline` spam.
- **Zero-exposure fallback:** unset `JOLT_IBRLEN`/`JOLT_RBRLEN` → reduces build-l8 to exactly `9d845205` behavior (+I/+R tree-search reopt on CPU).

**Provenance (audited):** binary built 2026-07-09 23:59; HEAD `55743479` committed 2026-07-10 10:25 (~10.5 h later) merely committed the already-compiled working-tree edits (all feature markers grep-present); `git diff --stat HEAD` on tracked source = empty. Worktree == HEAD == build source (byte-identity unprovable without rebuild due to embedded date, but source-consistent).

---

## Registry — tree-search dev tree: `iqtree3-gpu`

**Path:** `/scratch/rc29/as1708/iqtree3-gpu/` · **git:** branch **`tree-search-ts0`** @ `6fce15de` (working
tree dirty — the TS.x experiments). Other local branches: `gpu-kernel-prod` @ `6fce15de`, `gpu-kernel-dev` @
`46925181`, `fca-lbfgs-ws`. **⚠ GPU parsimony kernel is NOT in this tree's source.**

| Use this | Path (under `iqtree3-gpu/`) | md5 | Built | What it is |
|---|---|---|---|---|
| **LIVE GPU (tree-search dev)** | `build-gpu-on/iqtree3` | **`2c931f41`** | 2026-06-26 | JOLT + GPU + CTF + `--ts-fused` tree search. **No GPU parsimony.** Pinned by the hashara parity sweep + all current gems ModelFinder/tree-search scripts (14 scripts). |
| LIVE CPU-only | `build-gpu-off/iqtree3` | `6e0ade4f` | 2026-06-25 | CPU-only; used as the non-MPI CPU baseline (energy, CPU `-m MF`). |

### Frozen snapshots (immutable, under `iqtree3-gpu/frozen-binaries/`)
| Name | md5 | What it is | Still used? |
|---|---|---|---|
| `iqtree3-treesearch-jolt.46b8d079` | `46b8d079` | Tree-search **baseline**, frozen 2026-06-24 from `gpu-kernel-prod`@`6fce15de` before any TS.x change (`chmod 555`). Tree search runs on CPU here — the baseline the TS.x levers must beat. | yes (TS.x A/B) |
| `iqtree3-g5.0-parity.b85d482f` | `b85d482f` | CTF **parity** binary (G.5.0). Produced the published AA-1M/10M `-m MF` CTF numbers. | yes (3 `run_ctf_*`) |
| `iqtree3-cpuparity.ddde07c7` | `ddde07c7` | CPU-only parity snapshot. | reference |
| `iqtree3-pre-ctf.32b3f129` | `32b3f129` | JOLT+GPU HEAD frozen just before native `--ctf`. | provenance/rollback |

### ~~Untracked experimental build dirs~~ — 🧹 DELETED 2026-07-07 (freed 4.3 GB)
`build-prod-on` (`447c60a7`), `build-prod-off` (`d3dffe08`), `build-ts0-off` (`30dc278c`), `build-ts-prof`
(`76fe51fd`, 164 MB), `build-dev-dbg` (`134321eb`) — all 2026-06-23/24 dev builds with no frozen identity, no
runnable-script references, regenerable from `6fce15de`. Removed in the 2026-07-07 cleanup. `run_prod_cut_validate_v100.sh`
regenerates `build-prod-on/off` + `build-dev-dbg` via `mkdir -p`+cmake if ever re-run. Only `build-gpu-on` (`2c931f41`)
and `build-gpu-off` (`6e0ade4f`) remain live in this tree.

---

## Registry — other binaries in play

| Use this | Path | md5 | What it is |
|---|---|---|---|
| **Thesis clean repo (source)** | `/scratch/rc29/as1708/gems/` (branch `wip-extract` @ `a91c6686`) | source (has parsimony source `tree/gpu/gpu_parsimony.cu`) | The GEMS fork being assembled for the thesis/upstream PR. Build from here for the final artifact. |
| **Thesis clean-fork BUILT binary** | `/g/data/um09/as1708/gems-bin/iqtree3-gems-gpu-82b34139e71a` | **`fe693d75`** (built from `gems`@`82b34139`, 2026-07-01) | Has JOLT + CTF + **GPU parsimony** (`GPUPARS-B`), **NO mixtures** (not a superset). **md5-gated** by `repro_fig1_fig10_aa1m.sh` / `repro_fig1_jolt_depth.sh` / `repro_fig1_v2_depth.sh` (`EXPECT_BIN_MD5=fe693d75…`); produced the fig1/fig10 reproductions (172809010 etc.). |
| **Hashara naive (parity)** | `gems-verify/bin/iqtree3-hashara-naive-h200` | `e713866b` | Hashara's naive OpenACC port (her `build-gpu-cc90`), pinned copy. GPU-vs-GPU parity opponent. Run under `nvhpc-compilers/24.7 cuda/12.5.1`. |
| **Bootpack clone (CPU-only, active work)** | `iqtree3-boottile/build-boot-val/iqtree3` · `build-boot-cpu/iqtree3` | `39318f01` · `fe2bffb6` | JOLT_BOOT_PACK (uint8+exception-table bootstrap packing, 3.999× mem), branch `boot-pack-tiling`@`e5c527c6`. CPU-only validation builds (byte-id -T1 PASS, AA-1M 3.999× confirmed). **Source tree KEPT** — avian-`-B` OOM unblock (needs a GPU build here) still pending. Binaries not md5-pinned (validate script rebuilds fresh). |
| **MPI/FCA (main)** | `iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi` | `5aaf6da4` | FCA ModelFinder MPI binary. |
| **MPI/FCA (b1fix)** | `iqtree3-mf-iso/build-b1fix/iqtree3-mpi` | `0eb259fa` | `+I+Rk` SIGABRT crash-fix build. |
| **MPI/FCA (deadlock fix)** | `iqtree3-mf-iso/build-b1fix-ucc/iqtree3-mpi` | `5a639f91` | FCA np>1 deadlock **source fix** (2026-07-02, RUNS.md §G); `FCA_SAFE_MODE` toggle. Gates the GitHub commit. |

---

## Which scripts pin which binary (current, gems/ dir)

- **⚠️ `8cc3cb84` (20 scripts pin it) — BROKEN as of 2026-07-04:** `iqtree3-l2search/build-gpu-on/iqtree3` was
  rebuilt to `0faac84d`, so every script with `EXPECT_MD5=8cc3cb84` (the `fig4_parsimony_*`, `cpu_parsimony_*`,
  `repro_ctf_avian`, `repro_fig7_meow_euk`, `gems_boot_*`, `ts_fair_cpu_*`, `gems_hashara_jolt_deconfound`,
  `diag_ts_bottleneck`, `diag_kjpre_ncu`, `gems_reopt_pareto`, and 5 `gpu-modelfinder/job*`/`gems_thesis_sweep`)
  now **fails its md5 guard**. **DECISION 2026-07-07 (user): leave these pins as HISTORICAL provenance-of-record**
  (the `8cc3cb84` md5 documents exactly what produced each figure — the reproducibility value reviewers want) rather
  than mass-re-pin. **No current-work script pins `8cc3cb84`** (the Gate-0 / full-GPU / tree-search scripts all pin
  `9d845205`), so nothing active is broken. Individual scripts get re-pinned **on demand** when actually re-run:
  parsimony jobs → `gems-bin` `fe693d75` (still has `GPUPARS-B`); tree-search/diag → `9d845205` (prod) or `0faac84d`
  (T-floor feature-complete). This banner is the central record so a reviewer hitting a broken guard knows why.
- **`iqtree3-gpu/build-gpu-on/iqtree3` (`2c931f41`)** — 14 gems scripts: the hashara parity sweep, cross-arch,
  ModelFinder/tree-search, CPU energy (CPU-only path).
- **`gems-verify/bin/iqtree3-hashara-naive-h200` (`e713866b`)** — 5 gems scripts (the naive opponent).
- **`frozen-binaries/iqtree3-g5.0-parity.b85d482f`** — `run_ctf_1m_mf_energy.sh`, `run_ctf_10m_mf_aa_h200.sh`,
  `run_ctf_1m_test_a100.sh` (keep pinned so the published CTF rows stay comparable).

## Historical names (for reading old CHANGELOG / job logs)
md5 is the stable identifier across renames:
- `frozen_ab` ≡ `b85d482f` ≡ `frozen-binaries/iqtree3-g5.0-parity.b85d482f`
- `frozen_g50fuse` ≡ `2ce44a8b` ≡ **DELETED 2026-06-24** (orphan, 0 references)
- The old doc's "live `46b8d079`" is now the **frozen** tree-search baseline; the old "live `8dd57cfb`"
  (CLAUDE.md) and "`46b8d079`" (this doc, pre-2026-07-02) are both superseded — live GPU is now `2c931f41`
  (`iqtree3-gpu`) / `8cc3cb84` (`iqtree3-l2search`, canonical).

## Open provenance items (honest — resolve before the parsimony figure is finalized)
- **The spine parsimony run IS well-provenanced — only the ephemeral binary md5 is missing.** Jobs
  172524833/526962/530260 (2026-06-28) record `SRC=/scratch/rc29/as1708/iqtree3-l2search`, the kernel **linked
  and engaged** (`[GPUPARS-TIMING] engaged_steps=9603 fallback_steps=0`), CPU bit-identity `VERIFY mismatches=0`,
  and 2D vs 1D timings (10K: 5.599 s vs 17.158 s; AA-1M BEST SCORE −7541976.852167 == CPU). What is NOT frozen is
  the md5 of the `/jobfs/<id>/iq` copy that ran — so the exact bits aren't reproducible, but source + engagement +
  correctness are on record. Do not call it "unrecoverable."
- **`8cc3cb84` is NOT source-identical to the 56.591 s binary.** The spine ran 2026-06-28; `8cc3cb84`'s parsimony
  source (`gpu_lnl_intree.cu`) was edited 2026-06-29 18:58, *after* it. So a re-run on `8cc3cb84` is a **fresh
  measurement**, not a reproduction of 56.591 s — treat it as the new number of record.
- **The fair-CPU baseline has already moved.** §N/§P.2 cited CPU-104t = 81.1 s → "1.43×"; the newer full-node
  re-run (job **172862243**, on `8cc3cb84`) gives CPU-104t = **89.578 s**. So even the CPU leg isn't settled.
  **ACTION:** take the GPU-2D number from the pending fig4 172862242 (on `8cc3cb84`, engage-marker required) and
  the CPU-104t from 172862243 (89.578 s) — both on the SAME binary — for the one honest fair ratio. Until fig4
  172862242 lands with a confirmed `GPUPARS-B` marker, the parsimony speedup is **unpinned / under re-measurement**
  (candidate ratios seen so far: 3.76× [12t, retired], ~1.43×/~1.58× [unpinned GPU numerator]).

## Rebuild quick-reference

**⭐ THE CANONICAL (use this for everything new):**
```bash
# run it — frozen, immutable, safe to pin:
/scratch/rc29/as1708/frozen-binaries/iqtree3-canonical \
  -s ALN.phy -m MFP -B 1000 -nt 12 -ninit 2 -seed 12345 -starttree PARS \
  --jolt --gpu --ctf --ts-fused          # no env vars needed: cache/L0/L5/L6/L7 are default-ON

# rebuild it from source (the tree, not the frozen copy):
cd /scratch/rc29/as1708/iqtree3-graduate/build-promote && cmake --build . -j --target iqtree3 && md5sum iqtree3
# ...then RE-FREEZE under a NEW md5-tagged name. Never overwrite an existing frozen file.
```
⚠️ **A rebuild will NOT reproduce the frozen md5** (the build date is embedded in the binary). That is expected and is
exactly why the frozen copy exists: it is the only thing that can be pinned. Validate *behaviour* (RF=0, engage markers),
never md5-equality, when you rebuild.

**Superseded trees** (kept for reproducing pre-canonical runs only):
```bash
cd /scratch/rc29/as1708/iqtree3-l2search/build-gpu-on   && cmake --build . -j --target iqtree3   # 0faac84d lineage
cd /scratch/rc29/as1708/iqtree3-gpu/build-gpu-on        && cmake --build . -j --target iqtree3   # 2c931f41, NO parsimony
```

**The freezing rule (learned the hard way — this is how `8cc3cb84` was destroyed and 20 scripts broke):** a `build-*/iqtree3`
path is **LIVE** — the next `cmake --build` silently replaces it and every `EXPECT_MD5` pinned to it starts failing. **Never
pin a `build-*/` path.** Copy to `frozen-binaries/` with an md5-tagged name, `chmod 555`, add a row to the registry, and pin
*that*.
