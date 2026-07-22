# SPEC — `+I` mirror term for the clean-room whole-tree lnL (self-check / likelihood oracle)

> # ✅ IMPLEMENTED, SHIPPED & VALIDATED — do NOT re-implement (verified from disk 2026-07-18)
> This spec is **DONE**. The `gpuComputeTreeLnLCleanRoom` +I mirror described below is already in the merge line and
> the shipped binary — building it again would repeat the `+I+R`-duplicate mistake. Ground truth (merge tree
> `iqtree3-jolt-merge@30c0faf9`, re-grepped this pass):
> - **Kernel (Edit A):** `gpu_lnl_intree.cu:116` — `k1_node` root fold = `log(base_invar ? fabs(lh)+pinv*base_invar[ptn] : fabs(lh))`.
> - **Host (Edit E):** `phylotreegpu.cpp:107,131-142` — `gpuComputeTreeLnLCleanRoom` reads `pinv`, builds `base_invar`
>   (exact const_char/STATE_UNKNOWN/DNA|PROTEIN ambiguity), passes `pinv, base_invar.data()` to the crosscheck (`:225`).
>   The `:84` bail is **gone**. `launch_k1_node` / `gpu_lnl_crosscheck` carry `pinv, base_invar` params.
> - **Landed:** merge commit **`f413a891`** ("M.2 coverage closures: pure-invar"), an **ancestor of shipped `f3f7875f`**.
> - **Validated:** **invargate `173818786`** — `GTR+I` rel **5.95e-10** / `LG+I` **5.41e-10** / `GTR+I+R4` **0.00e+00**,
>   all **RF=0, GPU-declines=0, engaged** (declines=0 ⇒ the oracle engaged, NOT CPU-fallback-masked — the §5 worry is met).
> - **Flag:** merged as **inline default** (byte-identical when `pinv<=0`), NOT the `JOLT_NO_IMIRROR` kill-switch this spec proposed.
> - 🔴 **TRUE RESIDUAL (different axis — TREE SEARCH, not `-m MF`):** the +I bail still stands in
>   **`gpuComputeEdgeDervCleanRoom` (:431)** and **`gpuScreenNNICleanRoom` (:555)** — called from `phylotree.cpp:4456/4403`
>   (branch-derivative + NNI screener). So +I models fall to CPU only for tree-search *moves*. Separate, smaller task.
>   **The MF oracle this spec targeted is CLOSED.**
>
> 🔴 **THE RESIDUAL ABOVE IS HALF STALE — CORRECTED 2026-07-22 by source reading.**
> - **`gpuScreenNNICleanRoom` (`:562`) is NOT a production gap.** Its only callers (`phylotree.cpp:4403`/`:4456`)
>   are guarded by `params->ts_screen{,2,3}_check`, all of which **default false** (`tools.cpp:7871-7872`) — they
>   are `--ts-screen*-check` diagnostics. The **production** NNI screener is `gpuScreenNNIRank` (`:1519`, called
>   from `iqtree.cpp:4513`), whose own comment records the fix: *"(Was: `if (getPInvar()>0) return false` => CPU
>   fallback.)"* It now computes `base_invar` and passes `pinv`. **Delete the NNI half of this residual.**
> - **`gpuComputeEdgeDervCleanRoom` (`:436`) IS still a production gap** — reached via `computeLikelihoodDervGPU`
>   (`:2024`), installed at `:2088`; its NaN path delegates to `cpuComputeLikelihoodDervPointer`. Under test in
>   job `174344376` using the switch-free `[GPU-KERNEL]` / `[GPU-DERV]` markers.
>
> ⚠️ **POINTER REPAIRED 2026-07-22.** This line used to read "see WORK-LOG §4". The WORK-LOG was rewritten that day
> (2,739 → 230 lines; history at `research/archive/WORK-LOG-full-history-to-2026-07-22.md`) and its `§4` now means
> something else entirely, so the reference silently resolved to the wrong section. **The residual is now tracked in
> `WORK-LOG.md` §5 "Honest standing gaps" item 4**, re-verified open in source on that date
> (`phylotreegpu.cpp:436` and `:562`). Its priority ROSE: job `174338165` showed stock upstream selects `+I` in 9/9
> avian arms, so this CPU fallback hits the model class real data actually chooses — it is not a corner case.
> **Do not archive this file on the strength of its "SHIPPED & VALIDATED" banner; the banner covers the MF axis only.**

**Status (historic):** ready-to-implement spec. Read-only source recon complete 2026-07-16. ~~NOT built. NOT pushed.~~
→ **superseded: built + shipped + validated, see banner above.**
**Tree recon'd against:** `/scratch/rc29/as1708/iqtree3-pureinvar` (build-ctfab lineage, `eed14b92`).
All line numbers below are from this tree; re-grep before editing since they drift.
**Grounding rule:** every claim here traces to a source line read this pass; the WIN MAGNITUDE is
explicitly **unmeasured** and gated on the m4census result (job 174007767).

---

## 1. The defect (grounded)

`gpuComputeTreeLnLCleanRoom` (`tree/phylotreegpu.cpp:76`) is the reusable clean-room GPU whole-tree
log-likelihood. It is the **likelihood oracle** behind three GPU overrides of IQ-TREE's own likelihood
pointers:

| caller | line | role |
|---|---|---|
| `computeLikelihoodBranchGPU` | `:357` | **the per-evaluation likelihood during `-m MF`** (Brent α-tuning, final lnL). Returns the clean-room value directly to IQ-TREE. |
| `computeLikelihoodFromBufferGPU` | `:1997` | from-buffer whole-tree lnL |
| `-B` accepted-tree mirror (`saveCurrentTree`) | `:2461` | repopulates `_pattern_lh` for RELL bootstrap |

`gpuComputeTreeLnLCleanRoom` **unconditionally bails to CPU whenever `pinv > 0`**:

```
phylotreegpu.cpp:84 →  if (site_rate->getPInvar() > 0.0) return (double)NAN;   // +I omits ptn_invar in the clean-room sweep -> CPU
```

Consequence, in the codebase's own words at the `-B` call site:
```
phylotreegpu.cpp:2471 →  // gpuComputeTreeLnLCleanRoom declines pinv>0 (phylotreegpu.cpp:~80) => NaN => the CPU computeLikelihood() recovery below fires each such save.
```
So **every `+I` / `+I+G` / `+I+R` likelihood evaluation that routes through the oracle runs on CPU.**
This is the `-m MF` "CPU = declined `+I`" bucket (dnares 174003961 named `+I` decline as ~29 % of the
DNA-1M MF residual — the **largest *attackable* CPU bucket**; the larger ~44 % "graveyard driver storm"
was assessed non-attackable ⇒ Tier-2 dead).

> ⚠️ **This oracle decline is a DIFFERENT axis than the m4census (174007767) decline census.**
> m4census counts `[JOLT-GATE]` declines in **`optimizeParametersJOLT`** (`:2203-2211`) — a model
> falling *wholly* to CPU **optimization**. The mirror term fixes the **`gpuComputeTreeLnLCleanRoom`
> oracle** (`:84`) — a model whose optimizer runs on GPU but whose per-eval **likelihood** calls fall
> to CPU. A model can *pass* the optimizer gate and *still* hit the oracle decline on every evaluation.
> **m4census DNA arm (2026-07-16): `dna_on` = 0 pure-`+I` optimizer declines** (coverage working),
> `dna_off` = 1 forced (positive control live). That 0 means pure-`+I` DNA models *pass the optimizer*
> and therefore *do* reach the oracle — so the mirror term is exactly what they need — but m4census does
> **not** count oracle calls, so it does **not** size this win. The mirror-term wall impact is
> **unmeasured**; sizing it needs a dedicated before/after `+I`-model wall (§5).

**Root of the bail:** the kernel that computes the per-pattern likelihood, `k1_node` /
`k1_node_t<NS>`, ends the root fold with the *variable part only*:
```
gpu_lnl_intree.cu:87   (k1_node)      if (isRoot) patlh[ptn] = log(fabs(lh));
gpu_lnl_intree.cu:630  (k1_node_t)    if (isRoot) patlh[ptn] = log(fabs(lh));
```
For `RateGammaInvar`, `getProp(c)` sums to `(1-pinv)` and `getRate(c)` carries the `1/(1-pinv)`
rescale — both are read live at `phylotreegpu.cpp:101` — so `lh` already equals `(1-pinv)·V̄_p`
(the variable part). The **only** missing piece is the additive invariant term `pinv·base_invar[ptn]`.

---

## 2. The CPU bit-identity target (grounded)

CPU reference — `computeLikelihoodDervGenericSIMD`, `tree/phylotreesse.cpp:1196`:
```
1196  double lh_ptn = ptn_invar[ptn], df_ptn = 0.0, ddf_ptn = 0.0;   // starts with the invariant term
1199    lh_ptn += val0[i] * theta[i];                                // + variable part
1205  lh_ptn = fabs(lh_ptn);                                         // then |·|, then log-sum
```
where `ptn_invar[ptn] = pinv · base_invar[ptn]` (the `computePtnInvar` product, `phylotreesse.cpp:560`).

So CPU computes `log(fabs( pinv·base_invar[ptn] + V̄ ))`; GPU currently computes `log(fabs(V̄))`.
Adding `pinv·base_invar[ptn]` closes exactly the gap.

**Acceptance bar is `rel ≤ 1e-6`, NOT `5e-16`.** The clean-room oracle is *already* only a
`~1e-6` agreement with the CPU SIMD kernel (different summation structure — theta trick, per-category
fold; see `phylotreegpu.cpp:62`). The `+I` term is computed by the *identical* CPU formula, so the
residual after the fix is the pre-existing clean-room summation-order difference, unchanged. Safe by
construction: if the oracle is ever wrong, the `rel ≤ 1e-6` self-checks (`:2478`, and the standard
CPU-vs-GPU validation gate) fall back to CPU.

**Convention to match (already validated in this codebase):** the `+I` mix/derivative kernels use
`fabs(lh) + pinv*baseinvar[ptn]` (|·| on the variable part, then add) — e.g.
`k2_derv_mix_inv` `gpu_lnl_intree.cu:236`, `:428`, `:453`, `:481`. Use the same form here so the new
path is consistent with the four already-shipped `+I` insertions.

---

## 3. The fix — exact edits (all `pinv<=0` ⇒ byte-identical to today)

The `base_invar[ptn]` builder is **already replicated verbatim four times** in this tree — this is a
port of validated logic, not new math. Reference copies (identical `const_char` classification):
- `phylotreegpu.cpp:1636-1644` (A3 screener) ← simplest single-model copy, **copy this one**
- `phylotreegpu.cpp:1905-1918` (A1 all-branch-derv, per-class)
- `phylotreegpu.cpp:258-268` (A1 clsinv, per-class)
- `phylotreegpu.cpp:2307-2315` (optimizeParametersJOLT, with the canonical `computePtnInvar` comment citing `phylotreesse.cpp:560`)

### Edit A — kernels (`gpu_lnl_intree.cu`)
Add two trailing params `double pinv, const double* __restrict__ baseinvar` to **both**:
- `k1_node` (`:63`) and change `:87`:
  ```
  if (isRoot) patlh[ptn] = log( fabs(lh) + (pinv>0.0 ? pinv*baseinvar[ptn] : 0.0) );
  ```
- `k1_node_t<NS>` (`:602`) and change `:630` identically.

`pinv<=0` ⇒ `log(fabs(lh))` verbatim ⇒ **byte-identical** to today; `baseinvar` may be `nullptr`.

### Edit B — launch dispatcher `launch_k1_node` (`gpu_lnl_intree.cu:697`)
Add `double pinv, const double* baseinvar` to the signature; forward to all three kernel launches
(`:703`, `:704`, `:705`). Only the `isRoot=1` path dereferences `baseinvar`.

### Edit C — the six interior `launch_k1_node` callers (byte-identical)
The screener / all-branch-derv / reopt callers do interior folds only (`isRoot=0`, never log):
`:1759`, `:1866`, `:1876`, `:1878`, `:1971`, `:1977`, `:1992`. Pass `(0.0, nullptr)` — unchanged behaviour.

### Edit D — `gpu_lnl_crosscheck` launcher (`gpu_lnl_intree.cu:1249`)
- Signature (`:1249-1255`) + header decl (`gpu/gpu_iqtree.h:61`): add trailing
  `double pinv, const double* base_invar`.
- When `pinv>0`: alloc/upload a device buffer (pattern: reuse the `gb_baseinvar` DevBuf idiom at
  `:2240`/`:2361`, or add `gb_binv`); `cudaMemcpy` the `nptn` doubles H2D once.
- The `:1289` `launch_k1_node` call passes `(pinv, d_binv)` — the kernel uses them only at the root
  descriptor (`isRoot`), so all interior folds in the same loop ignore them.
- `pinv<=0` ⇒ no alloc, pass `(0.0, nullptr)` ⇒ byte-identical.

### Edit E — host wrapper `gpuComputeTreeLnLCleanRoom` (`phylotreegpu.cpp:76`)
- **Delete the bail at `:84`.**
- After `freq` is populated (`:97`), add:
  ```
  double pinv = site_rate->getPInvar();
  vector<double> base_invar;                       // built only when +I
  if (pinv > 0.0) { base_invar.assign(nptn,0.0); /* copy the :1636-1644 classification */ }
  ```
- Pass `pinv` and `(pinv>0.0 ? base_invar.data() : nullptr)` into the `gpu_lnl_crosscheck` call
  at `:178`.

**No `(1-pinv)` rescale needed here** — `catRate`/`catProp` are read live from `site_rate`
(`:101`), so the rescale is already baked in (matches A3 comment `:220-222`, A1 `:1490-1491`).

---

## 4. What this does — and does NOT — fix

**Fixes:** the whole-tree lnL **oracle** (`gpuComputeTreeLnLCleanRoom`) for `+I`/`+I+G`/`+I+R`, so
`computeLikelihoodBranchGPU` (per-eval MF likelihood), `computeLikelihoodFromBufferGPU`, and the `-B`
accepted-save mirror stay on GPU instead of falling to CPU. It evaluates lnL *at whatever pinv the
optimizer already chose*.

**Does NOT fix:** the `+I+R` **optimizer** (`gpu_jolt_optimize`, the JOLT joint LM). That path has its
own, separate `+I` machinery (`applyPinv`, base_invar upload `:2740`, the `kj_derv_fused` `+I` variants)
and the already-graduated FD-normalization fix **`JOLT_IR_FDFIX`** (job 173898475). This mirror term is
orthogonal to that — do not conflate them.

---

## 5. Flag & validation gate

- **Flag:** default-ON kill-switch `JOLT_NO_IMIRROR` (mirrors the `JOLT_NO_PUREINVAR` /
  `JOLT_NO_FIXINVAR` convention). `JOLT_NO_IMIRROR` set ⇒ restore the `:84` bail (old behaviour).
  Rationale for default-ON: safe-by-construction (`pinv<=0` byte-identical) and self-check-gated.
- **Proof-of-build:** `strings <bin> | grep JOLT_NO_IMIRROR` present; and a measured effect — the
  `[GPU-BRANCH]` engage marker firing on a `+I` model where today it declines.
- **Correctness gate (the real bar):** one `+I+G` model each side of the state-count axis —
  DNA `GTR+I+G4` and AA `LG+I+G4` — GPU (mirror ON) vs CPU-only, **`rel ≤ 1e-6` on final lnL and
  RF = 0**. Then a `+I+R` cell (`GTR+I+R4`) for the ladder. Reuse the existing `-m <model> --jolt --gpu`
  harness + a `--nogpu`/CPU reference, same shape as every prior `+I` gate.
- **No wall-time claim** until the gate is green AND a dedicated before/after wall on a `+I` model
  (mirror ON vs `JOLT_NO_IMIRROR`, same `-m <+I model> --jolt --gpu -nt N`) sizes the saved CPU
  fallback. m4census does **not** size this (it counts optimizer-gate declines, a different axis — §1).

---

## 6. Effort & risk

- **Effort:** ~half a day to implement (6 localized edits, all following 4 existing precedents) + one
  short validation job. The single a-priori risk (base_invar semantics / ambiguity buckets) is already
  answered — copy the validated `:1636-1644` replica.
- **Risk:** LOW. `pinv<=0` byte-identical; self-check-gated; exact CPU-formula reuse; no new kernel,
  no new math, no change to the `+I` *optimizer*.
- **Honest caveat:** this removes a CPU fallback; it is **not** proven to be a net `-m MF` speedup until
  a real before/after `+I`-model wall is measured (§5). m4census does **not** size it — it counts a
  different decline axis (optimizer gate, not the oracle; see §1). The plan doc's "#1 faster lever"
  ranking means *largest attackable CPU bucket*, not a measured win.
