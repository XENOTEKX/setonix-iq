# CONVERGED ROOT-CAUSE + PROPER-FIX DESIGN — two open risks

**Status:** investigated (2 agents) + red-teamed (1 agent) + source-verified by hand, 2026-07-16. **No source changed, nothing built, nothing pushed.** Both fixes need ONE cheap GPU experiment before building (GPU busy with `174010664`).
**Method:** every load-bearing claim traces to a file:line or job log. My own earlier overclaims are RETRACTED in place (marked 🔁).

---

## ISSUE A — the >12h AA-1M full-MFP run (job 173919739, binary `bffdc16e`)

> # 🔴🔴 REVERSED — 2026-07-17, gate C (job `174030908`) + AA-1M reward (`174036432`)
>
> **The release-hook fix (`jolt_gpu_release_pools`) is EXACT but delivers ZERO measured tree-search speedup on
> the shipping merge binary (`a0eeb734`/`f3f7875f`) on the H200. The RTILE fix (2026-07-15, already in the
> merge) already solved the VRAM starvation.** Everything below describing the balloon as a live tree-search
> defect was TRUE for `bffdc16e` (old unconditional `if(freeRate)nTile=1` pin) but does NOT transfer to the
> RTILE'd merge binary — the exact "check the artifact on disk, not the story" trap.
>
> **The A/B (release ON vs `JOLT_NO_POOL_RELEASE=1`, ONE binary, within-job):**
> ```
>   DNA-1M full -m MFP:  ON nTile=1 / OFF nTile=1  → TIE   (treefile+lnL bit-identical d1b14e2d)
>   AA-1M -m MFP -mset LG: ON freed 120.75GB nTile=13 57m8s / OFF nTile=12 54m43s → TIE (OFF faster; lnL id.)
> ```
> **Why it does nothing:** both arms settle at **~17 GB free** regardless of the 120 GB balloon — tree search's
> OWN grow-only pools (`gb_upper` etc.) re-fill whatever is free, landing at the same nTile. The RTILE fix caps
> the +R MF balloon so freeVRAM bottoms at ~18 GB (nTile≈12, ~55min) instead of `bffdc16e`'s 0.4 GB (nTile=604,
> >12h). **Gate A's 138.7→0.4 GB / nTile=604 was `bffdc16e`-ONLY.**
>
> **DISPOSITION (user-approved): keep the release-hook as labeled OOM-INSURANCE only** (exact, free; headroom
> for LARGER-than-1M full-MFP e.g. avian TENT, or A100 80GB). Branch `iqtree3-vramrelease`@`vram-pool-release`.
> **NOT frozen/registered/pushed as a performance fix.** The flagship `9d845205` still has the old pin but only
> runs fixed-model `-m LG+G4` (MF skipped) so never triggers; if ever full-MFP-large, backport RTILE not this.
>
> The bit-identity correctness result IS solid and reusable (release is byte-invariant, 52 pools written-before-read).

### Root cause — CONFIRMED (source-verified + red-teamed)
Grow-only, **never-freed** JOLT device pools. `devbuf_ensure` (`iqtree3-graduate/tree/gpu/gpu_lnl_intree.cu:805-810`) only `cudaFree`s to **grow** (`:807`); "Never freed (released at process exit)" (`:803`). During ModelFinder the pools reach a high-water-mark at **`LG+R5`**: `if (freeRate) nTile = 1;` (`:2685`) pins every +R model to the **full 946,439-pattern width** ⇒ `gbj_partial` ~69 GB + `gbj_prepool` ~11 GB ≈ **82 GB in one model**. freeVRAM collapses 138.7 → **0.4 GB by MF model ~#8** (`prim.console:202`) and stays pinned for the rest of MF **and the whole tree search**. Tree search (which needs only `LG+G4`, ncat=4) is starved ⇒ `mix_pick_ntile` (`:2680-2682`, `nTile = ceil(foot/(0.8·free))`) returns **nTile=604** ⇒ **~330 s/iteration** ⇒ ~13.5 h projected ⇒ manually SIGTERM'd (exit 271) at 12h16m.

**NOT** a leak / hang / OOM-crash / livelock — a steady plateau (nTile=604 constant, never climbs), and a single `BETTER TREE FOUND` at iteration 1 is normal stochastic-NNI behaviour. GPU==CPU at iter 1 (`rel=7.6e-14`).

🔁 **RETRACTED (my earlier claim):** the "111.70 GB *resident* upper pool" was WRONG — `upperGB(nt=1)=111.70` is a printed *what-if* for `gb_upper` at nTile=1, not the actual residency. The real filler is the grow-only optimizer pools above.

**Stale rationale:** the `if (freeRate) nTile=1` comment ("+R declines to CPU") is **FALSE** — the L5 graduation runs +R on the GPU (`LG+R5` did 401 GPU joint-iters, `prim.console:199`), which is exactly what allocates the 82 GB.

### The confound — CONFIRMED (this is a latent defect in the SHIPPING tree)
The promoted `9d845205` (`iqtree3-l2search-stage2b`, HEAD `eef09e2c`) has the **byte-identical** defect: grow-only `devbuf_ensure`, `if (freeRate) nTile=1` (`:2691`), **no release hook**. Every healthy AA-1M run used `-m LG+G4 -B` (**MF skipped**) ⇒ `LG+R5` never ran ⇒ pool never ballooned. **No promoted full-MFP-large run exists that didn't collapse.** So the promoted binary does not "do it right" — it dodged the trigger. This bug bites any full-MFP run on a large alignment.

### The proper fix — release the MF-phase pools at the verified seam (handles the intermediate phase)
**Seam (hand-verified in `main/phyloanalysis.cpp`):**
- `startTreeReconstruction` (`:3406`) does: `computeInitialDist` (`:3439`, pre-MF ML distances) → `runModelFinder` (`:3461`) → `runMixtureFinder` (`:3464`).
- `runTreeReconstruction` (`:3521`) does: **`computeInitialDist` again (`:3539`, post-model ML distances + param init)** → candidate-set init → tree search.
- Main flow: `:5742 startTreeReconstruction` → `:5743 runTreeReconstruction`.

**⇒ Release site = END of `startTreeReconstruction` (after `:3464`, before return).** This is the ONLY correct point:
- NOT at `:3461` (too early — `runMixtureFinder :3464` re-grows).
- NOT at the top of `runTreeReconstruction` (churns — it is re-called per **legacy `-b`** replicate, `:4682/:4748/:4780`, with no preceding MF).
- At end of `startTreeReconstruction` it fires **exactly once per analysis**, after all MF/mixture GPU work, and — **this is your intermediate-phase point** — *before* `runTreeReconstruction`'s `computeInitialDist` (`:3539`) + param-init, so **both the intermediate ML-distance phase AND tree search inherit the freed VRAM** and re-grow only what they need.

**Mechanism:** `extern "C" void jolt_gpu_release_pools()` that `cudaFree`s the MF-grown DevBufs (`gbj_partial, gbj_prepool, gbj_pretmp, gbj_tipeig, gbj_echild`, the `gb_m*` mixture pools, screener `gb_partial/gb_upper`) and resets each `cap=0`. `devbuf_ensure` re-grows on demand.

**Statelessness — SAFE (red-team-verified):** every pool is written-before-read each call (the `:800-803` contract: "fully overwritten every call → only the ALLOCATION persists"); freeing + re-growing cannot change any result. `gb_upper`'s "persistent" upper is *within-call* (root-children seeded by `k1_node_prod` `:2105-2107` before any child reads it).

### 🔁 HONEST quantification — my "nTile=1 / 45 s per iter" was ARITHMETICALLY FALSE (red-team)
The tree-search **screener** foot for AA-1M `LG+G4` is **169.82 GB at nTile=1** (`gb_upper` = 198·80·946439·8 = **111.7 GB** + lowers 55.3 GB — a FIXED property of this data/model). Max free after full release ≈ 138.7 GB ⇒ budget 0.8·138.7 = 111 GB ⇒ **nTile = ceil(169.82/111) = 2, never 1.** The healthy `-m LG+G4 -B` run measures **~62 s/iter** (not 45), and it too is nTile≥2 (same 169.82 GB foot). **So the fix recovers ~5× (nTile 604→2-3, ~330→~62 s/iter) — real and worth it — but the honest target is nTile=2-3 / ~62 s/iter. Do NOT quote nTile=1 / 45 s.**

### Deeper root (the release is a band-aid over two deeper issues — flagged, not required)
1. **Non-shrinking allocator** — `devbuf_ensure` never shrinks. A generic shrink-on-demand / phase-arena fixes THIS and every future high-water case, but naive shrink thrashes (free+realloc per call) ⇒ needs phase-awareness ⇒ the explicit phase-boundary release is the pragmatic version.
2. **Stale `freeRate` pin** — `if (freeRate) nTile=1` (`:2685`) is what balloons +R in the first place. Making the +R MF path tile-aware (never allocate full-width) is the deepest fix, but the +R optimize loop is not tile-aware (`setChunk(0)` `:2839` assumes nTile==1) ⇒ real work. At minimum, reconcile the stale comment.

### GATE A — ✅ BUILT + VALIDATED: `gadi-ci/gems/gems_gate_a_mfp_vram.sh` (needs NO source change, NO build)
**The insight that makes it free:** a `-m LG+G4` run **skips ModelFinder**, so its pools never balloon and tree search sees **full VRAM** — i.e. **`-m LG+G4` IS EXACTLY THE POST-RELEASE STATE.** So the fix's reward is measurable *before* implementing it. Same binary (`bffdc16e`, the 12h run's own binary) both arms ⇒ the only variable is "did MF run".
- **A1** = full MFP, time-boxed 60 min (the collapse is early — MF model ~#8, `prim.console:202` of 3153) ⇒ confirm freeVRAM **138.7 → ≤2 GB** live, at a +R model.
- **A2** = `-m LG+G4`, time-boxed 60 min ⇒ measure tree-search **nTile** + per-iter at full VRAM = the post-release state.
- **A2 is FALSIFIABLE and settles the dispute:** if measured **nTile==1**, the red-team's arithmetic is WRONG and my "nTile=1 / 45 s" stands; if **nTile≥2**, the correction is right and the honest reward is 604→2-3. **No reward number goes in any doc until this arm prints one.**
- Liveness-gated (an arm with no telemetry ⇒ **VOID**, never a silent pass — the rfavor-euk lesson). ~2 h, 1 GPU. Validated: `bash -n` clean, md5 `bffdc16e` asserted.

---

## ISSUE B — the AA `+I+R` 0.07-nat gap ("④")

> # 🔴🔴 REFUTED IN FULL — 2026-07-17, Gate B v3, job `174020222`
>
> **THE GAP THIS ENTIRE SECTION EXPLAINS DOES NOT EXIST.** Everything below was reasoning about a
> **GPU-vs-CPU deficit that was never measured against CPU.** Kept visible (not deleted) because the
> *mechanism* of the error is the lesson.
>
> **The first honest `--no-jolt` CPU MLE** (this section, and every job it cites, never ran one — see §7a
> defect #6: a bare invocation runs **GPU JOLT by default**, `main.cpp:2289`):
>
> ```
>   CPU alone   −7541972.885   324 s    ← the TRUE CPU MLE. WORST of the three.
>   GPU JOLT    −7541972.346     6 s    ← +0.539 nats BETTER than CPU, ~54× faster
>   CPU-RESCUE  −7541972.276            ← the ".276 CPU MLE" quoted below. NOT a CPU result:
>                                          it is the JOLT state + a CPU polish (FDFIX-OFF ⇒
>                                          write-back MISMATCH ⇒ CPU fallback). CPU alone
>                                          cannot reach it — it lands 0.61 nats worse.
> ```
>
> **Consequences:**
> - **The shared-`mu` ratchet cannot be the mechanism of a GPU deficit — there is no deficit.** The argument
>   is self-refuting: **a pure-CPU run has no `mu` at all, and does 0.539 WORSE.** (The ratchet may still be
>   real *as optimizer behaviour*; it explains nothing about a GPU-vs-CPU gap.)
> - **`JOLT_IR_BESTWB` is a confirmed NO-OP** — `gpubwb` was lnL-**identical** to `gpubase`, closing the last
>   unmeasured combination (FDFIX-**ON**+BESTWB-ON). Its real `-te` value is **−7541972.345**, and the
>   "reaches −7541972.270" claim below is **my conflation of an `-m MF` scan-table row**
>   (`mf_aa_bestwb.console:209`) with a `-te` MLE. **BESTWB is finished. Do not ship or re-test it.**
> - **OPG/empirical-Fisher is now aimed at a non-existent defect.** Already deferred; now moot.
>
> **What actually survives:** the rescue reaches **+0.07 over JOLT** ⇒ a possible **post-JOLT CPU polish
> OPPORTUNITY**, not a defect. At **0.14 BIC** it is selection-irrelevant. Build only against a real
> <0.5-BIC `+I+R` tie.
>
> ⚠️ **SCOPE:** ONE cell (AA-100k, `LG+I+R4`, seed 12345, fixed `-te` tree) — the exact cell ④ was argued on,
> so it settles ④. It does **NOT** license a general *"JOLT beats the CPU optimiser"* claim. **Do not quote
> the 0.539 or the 54× in the thesis or any Hashara-facing doc on the strength of one cell.**

### ~~Root cause — CONFIRMED (source-verified + red-teamed)~~ → mechanism of a gap that isn't there
Single **shared Levenberg `mu`** across all arms (brlen, alpha, pinv, free-Q, +R rates/weights) in the freeRate backtrack loop (`iqtree3-mfdevcheck/tree/gpu/gpu_lnl_intree.cu:3306-3349`; `mu=1.0` `:3169`; `muIn=mu` at iteration entry `:3271`; accept `mu*=0.5` `:3343`; reject `mu*=4` `:3349`). The ×4-up/×0.5-down asymmetry lets `mu` **ratchet to ~1e9** (convtrace job 173907382: it=4 rej=10 mu→6.5e4; it=16 mu=1e9 `exit=CONV(dl<tol)` at −7541972.346). Once `mu ≫ |dd|`, every step collapses to `grad/1e9` ⇒ premature `dl<tol` exit **0.07 below the optimum**.

### Scope — verified, and 🔁 my "overturns the doc" framing RETRACTED
🔴 **"vs CPU −7541972.276" IS FALSE — that is the CPU-RESCUE value, not CPU** (job `174020222`: true CPU = **−7541972.885**, i.e. the GPU is **0.539 BETTER**, not 0.070 worse). Numbers as originally written (jobs 173898475 / 173910972 / 173907382): AA `LG+I+R4` GPU **−7541972.346** vs ~~CPU~~ **rescue −7541972.276**; DNA `GTR/F81/JC +I+R4` reach the **exact** CPU MLE; pure AA `LG+R4` byte-exact. Scope = **AA / high-state + the +I pinv flat ridge** (the condition-number spread between AA's sharp arms and the near-flat pinv ridge — `pinvMax≈0.028`, opt drives pinv→~1e-6 — is too large for one shared `mu`). DNA = 0.

🔁 **RETRACTED:** I framed this as "overturns `MF-FULL-GPU-COVERAGE §3c` (nFreeQ==0)." That was a **conflation** — §3c's "nFreeQ==0 trigger" is about the **already-fixed ~10-nat props-deficit bug** (`JOLT_IR_FDFIX`), a *different* bug; the doc **already** attributes the residual 0.07 to AA (§3c). So this **confirms** the doc. The surviving substantive point: **don't gate the ④ fix on `nFreeQ==0`** (that would floor clean fixed-Q DNA and caused the R4 regression, `IR4_REDBLUE:82`).

**CPU isn't at the true optimum either** (same-tree bracket): `frzbrlen` = −7541972.236 > CPU −.276 > GPU −.346, so BOTH optimizers strand below the `+I+R` optimum (≥ −.236). (The "pure `LG+R4` −.238 beats CPU" citation mixes trees — loose; the `frzbrlen` bracket is the rigorous evidence.)

### The proper "fix" — HONEST verdict: do NOT build OPG-Fisher now
- **OPG/empirical-Fisher** (the doc's principled parameter-free fix) is **UNBUILT, design-only, and EXPLICITLY DEFERRED by a prior user decision** (`JOLT-OPG-FISHER-OPTIMIZER.md`: "BUILD DEFERRED"; `curvFloor` was **reverted** after regressing DNA + R8/R10). Building an optimizer redesign for a **0.14-BIC, selection-irrelevant** gap (real-data selection margins are 2.7–1959 BIC; the real-data winner DNA `GTR+F+I+R4` has **zero** gap) is disproportionate.
- **`muIn`-decay** (on accept reset `mu=fmax(muIn*0.5,…)` from the iteration-**entry** `mu`, not the inflated post-backtrack value) is a REAL one-line mechanism fix that severs the **cross-iteration** ratchet — but **not proven** to close the gap (the `dl<tol` premature exit survives; a flat-ridge step can still be tiny). Unproven, cheap to test.
- 🔴 ~~**`JOLT_IR_BESTWB_EN`** (②a snapshot restore, `:3346`) **already exists** and reaches −7541972.270 (closes MOST of the gap) ⇒ much of the gap is **reject-EXIT state-restore**, not fundamental curvature. Cheaper than OPG, already built.~~ **REFUTED TWICE OVER** (job `174020222`): (a) the env is `JOLT_IR_BESTWB`, not `_EN`; (b) **−7541972.270 is an `-m MF` scan-table row** (`mf_aa_bestwb.console:209`), **not a `-te` MLE** — my conflation. BESTWB's real `-te` value is **−7541972.345**, and with FDFIX-ON it is **lnL-IDENTICAL to base** ⇒ **a confirmed NO-OP**. It closes nothing because **there is nothing to close.**

### GATE B — ✅ BUILT + VALIDATED: `gadi-ci/gems/gems_gate_b_ir4_mle.sh` (needs NO source change, NO build)
Decides the **fix CLASS** — is the 0.07 *cheap state-restore* or *fundamental curvature*? 3 arms, ONE binary (`build-irbestwb`, md5 **`50f53ca1`**, BESTWB=2 + FDFIX=3 verified), AA-**100k** (the exact ④ baseline scale) + the exact ④ `-te` tree (`iplus_173879879/aa.treefile`):
`cpuref` (CPU reference MLE, **measured fresh — not quoted**) · `gpubase` (BESTWB OFF ⇒ expect the ~0.07 shortfall) · `gpubwb` (`JOLT_IR_BESTWB_EN=1`).
- **BESTWB reaches ≥ cpuref** ⇒ the gap is the **reject-EXIT state-restore**, not curvature ⇒ **a cheap real fix exists and OPG is never needed** (next: harden the restore — red-team: BESTWB solo is INCOMPLETE, it skips the zero-accept exit).
- **BESTWB does NOT close it** ⇒ the gap IS the shared-`mu` curvature failure ⇒ **ACCEPT it** (0.14 BIC, selection-irrelevant); do NOT build OPG unless a real AA dataset shows a <0.5-BIC `+I+R` tie.
- Guards: proof-of-build sentinels asserted; **inconclusive branch** if `gpubase` fails to reproduce the ~0.07 (⇒ baseline mismatch, NOT a pass); liveness ⇒ VOID on any empty arm. ~1 h, 1 GPU. Validated (`bash -n` clean; verdict logic dry-run against the documented numbers → correct branch).
- 🔴 **Finding while building it:** **`JOLT_IR_MUIN` does not exist in ANY built binary** (`MUIN=0` across all 12 `build-*/iqtree3`) ⇒ the muIn-decay lever needs a **BUILD** and is deliberately **excluded** from this gate. Only run it if Gate B says "curvature" *and* someone still wants to chase 0.14 BIC.
- ⚠️ **Diagnostic, not a ship decision.** Any shipped change still needs its own rel≤1e-6 + selection-unchanged + DNA-no-regression gate.

### Converged recommendation (Issue B)
1. Root cause documented (above) — the shared-`mu` ratchet, verified.
2. Run **Gate B** (above) to decide the fix class. Do nothing else first.
3. Do NOT build OPG-Fisher unless Gate B says "curvature" **and** a real AA dataset ever shows a <0.5-BIC `+I+R` tie.

---

## Summary of the two proper fixes

| | Issue A (>12h MFP) | Issue B (0.07-nat ④) |
|---|---|---|
| **root cause** | grow-only never-freed pools balloon at LG+R5 (stale `freeRate` pin) → starve tree-search tiler | shared-`mu` Levenberg ratchet → mu~1e9 → step collapse → premature `dl<tol` |
| **verified** | source + confound + arithmetic (red-team) | source + convtrace + scope numbers (red-team) |
| **proper fix** | release MF pools at end of `startTreeReconstruction` (:3464) — handles the intermediate distance/param phase | mechanism fix (muIn-decay / BESTWB) if cheap A/B closes it; else accept — OPG deferred |
| **honest reward** | **~5×** (nTile 604→2-3, ~330→~62 s/iter) — NOT nTile=1/45s | 0.14 BIC, **selection-irrelevant**; real-data winner has zero gap |
| **priority** | **real** (latent in the shipping tree; any full-MFP-large run) | **low** (build only if a real AA tie ever appears) |
| **gate before build** | ✅ **BUILT:** `gems_gate_a_mfp_vram.sh` — A1 MFP (balloon) vs A2 `-m LG+G4` (= post-release state). **No source change needed.** Falsifiable: prints the measured nTile. | ✅ **BUILT:** `gems_gate_b_ir4_mle.sh` — cpuref / gpubase / `BESTWB_EN=1` on the exact ④ `-te` case. **No build needed.** Decides fix CLASS. |
