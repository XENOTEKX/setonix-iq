# OPG conditioned +R optimizer — implementation design (build spec)

**Status: PHASE 1 PASSED + committed `1bb82e14`; PHASE 2 IMPLEMENTED + gate submitted (job `174158000`), AWAITING RESULT
(2026-07-20).** Foundation = GATE-0.5 FULLY CLEARED (both per-category gradients FD-validated, pinv=0 AND +I+R,
DNA+AA+real-avian, job `174154357`). Both Phase-2 audits returned **REVISE-FIRST**; every finding folded into **§13 (v2
build spec)** and **§14 (as-built)**. Author-run builds only; **no GPU-source push**. Companions:
`FREERATE-CONDITIONING-AND-IDENTIFIABILITY.md` (investigation/gates), `JOLT-OPG-FISHER-OPTIMIZER.md` (prior art, re-targeted).

🔴 **LINE-REF PROVENANCE:** §0–§9 line refs are the OLD `iqtree3-rgradrate` tree (design-era). **The BUILD tree is
`/scratch/rc29/as1708/iqtree3-opg`; §10–§14 use its refs (verified on disk).** When implementing, follow §13/§14, not the
§0–§9 numbers. **Build in the worktree; never edit canonical.**

---

## 0. What we replace and why

The +R joint LM diagonal backtracking loop (**:3480-3524**) steps each rate/weight coordinate independently on a noisy FD
secant: **:3492** `ny_c=baseY[c]+g_y[c]/(|ddY[c]|+mu)`, **:3494** `cz_c=baseZ[c]+g_z[c]/(|ddZ[c]|+mu)` (secants :3435-3437,
`mu` ratchet ×4 :3524 / ×0.5 floor 1e-9 :3518). When `k > true rate classes` the degeneracy is **off-diagonal** ⇒ ρ≈0.975
crawl → 401 cap; flat valley ⇒ seed-dependent stop → RF 4,0,4 at tol 1e-2. **OPG replaces ONLY this (y,z) block.** brlen
stays on `g_df/g_ddf` (:3481); alpha/pinv/Q keep diagonal arms; `gaugeFix` (:3193) on accept.

Two mechanisms the audits surfaced that OPG removes *by construction* (record them — they strengthen the case):
`gaugeFix` shifts every `baseY` by `−log m` on accept, **polluting the ddY/ddZ secants** (:3435-3437); and the `mu→1e9`
ratchet makes steps tiny so `dl<tol` (:3523) can fire as a **false convergence**.

---

## 1. Per-pattern scores `s_p` — VALIDATED (re-derived independently by BOTH audits)

Per-SITE scores `s_p=∂logL_p/∂θ` (no ptn_freq; the Gram applies freq once). All ingredients resident after
`computeGradient`'s per-chunk work — **no new likelihood pass**.
```
RATE   y_c=log(meanR_c):   s^y_{p,c} = catRate[c]·catProp_v[c]·invl[p]·rnum[c][p]
WEIGHT z_c (softmax logit): r_c(p)=invl[p]·wnum[c][p];  R_p=Σ_k r_k(p);  s^z_{p,c} = r_c(p) − bprop[c]·R_p
```
**Why it is the TRUE per-site derivative (not a rearrangement):** `rnum[c][p] += g_rscale[c]·rc` (:479) with
`g_rscale[c]=bv/(catRate[c]·catProp_v[c])` (:3053) — the prefactor cancels that denominator exactly, leaving
`s^y_{p,c}=invl[p]·Σ_edges b_e·rc_{e,c}(p) = ∂logL_p/∂y_c` (rates enter only as `len=brlen·catRate`, :2860). `rnum` is
`[c][p]`, summed over **edges only**, never patterns ⇒ genuinely per-pattern. `accR` carries **no** `catProp_v` (:811-823),
which is applied exactly once at :3143 and `catRate` once at :3434 ⇒ **no double-count**. Softmax chain rule
`∂w_c/∂z_j=w_c(δ−w_j)` gives `s^z` exactly. Under +I, `R_p=1−pinv·I_p/L_p` and `Σ_p freq·R_p=sumWN` — reconciling with the
`wnorm=optPinv?sumWN:rN` selector (:3149). Both reductions check out: `Σ_p freq·s^y=g_y`, `Σ_p freq·s^z=gzR`. ✓

🔴 **Bitwise caveat (blue-team, would false-fail the gate):** at pinv=0 the device gradient uses `wnorm=rN` (:3149) while
per-pattern `R_p` reconstructs `sumWN`. Equal mathematically, **not bitwise** ⇒ the host reference MUST compare against the
**sumWN form**, else a ~1e-12·rN ≈ 3.7e-5 absolute false-fail at avian scale.

🔴 **Prior-art corrections (disk wins):** the Gram is **NOT "free"** (`kj_reduce_gradnum` is a per-category reduce; the Gram
is outer-product-**before**-reduce = a genuinely new kernel); the per-pattern z-score must use true `R_p`, not the `rN`
constant (absent from prior art — the #1 bug-hiding spot); prior-art line refs are stale.

---

## 2. Insertion points (complete — audits added C2/I2)

| # | site | action |
|---|---|---|
| A | `gpu_lnl_intree.cu` ~**:806-823** | ADD templated kernel `kj_opg_gram` + stage-2 reduce (§3) |
| B | before `gpu_jolt_optimize` (:2626) | ADD static host `jolt_jacobi_eig()` (self-contained; §5) |
| C | ~**:2653** (env block) | `JOLT_OPG` master + `JOLT_NO_OPG` kill-switch + `JOLT_OPG_{GRAMCHECK,DIAG,DENSE,LMIN,PROJY}`; **all default-OFF ⇒ byte-identical** |
| D | ~**:2822-2824** (`freeRate==1` alloc) | ADD `gbj_opg` DevBuf `[Nch*GBmax]` + host mirror — **gate the ALLOCATION on the flag**, not just the launch (never-freed pools :836-841; open MFP VRAM-balloon defect) |
| E | `computeGradient` chunk loop ~**:3137-3141** | gated `kj_opg_gram` launch + stage-2 + D2H + per-chunk Kahan into `H`. **The ONLY point all four inputs are valid** (both audits confirm: `d_rnum` memset :3037, complete after `proc(root,-1)` :3124; `d_wnum` :3105-3117 & `d_invLbase` :3100 at the chunk's first edge) |
| F | ~**:3489-3499** | REPLACE the two diagonal lines with the OPG δ (§4); OFF ⇒ unchanged |
| G | ~**:3518/:3524** | OPG branch: Nielsen ρ updates λ only (§4, H12); diagonal path keeps mu |
| H | after LM loop ~**:3540-3559** | gated λ̂_min on the **(2k−2) diagnostic** matrix (§5) → `*out_lambda_min` |
| I | signature **:2635-2637** | ADD `double* out_lambda_min` (nullable last param) |
| I2 | `tree/gpu/gpu_iqtree.h:238` **+ `phylotreegpu.cpp:2411` AND `:2914`** | header + **BOTH** call sites (`:2914` = the `maxiter=0` DEVCHECK eval; missing it = compile error) |

---

## 3. `kj_opg_gram` — the actual kernel spec (was prose; blue-team H1 blocker)

`Nch=k(2k+1)` dense upper-tri (`2k` diagonal-only mode). **Cost reality (blue-team, measured-by-estimate):** at ncat=10,
Pn=1e6 the Gram is ~160 MB traffic (~55 µs) + 2.1e8 FMA (~12 µs) against a ~40 ms gradient sweep ⇒ **0.2–0.35 %**. Red's
"21× the reduce work" ratio is right but attached to a **non-risk**; the real cost risk is H2 below.

```cuda
template<int K>                                   // K=ncat, instantiated 2..10
__launch_bounds__(256,4)
__global__ void kj_opg_gram(int Pn,               // 🔴 H7: chunk width, NOT host nptn
        const double* __restrict__ rnum, const double* __restrict__ wnum,
        const double* __restrict__ invl, const double* __restrict__ ptnfreq,
        const double* __restrict__ cRcP,          // catRate[c]*catProp_v[c], K doubles
        const double* __restrict__ bprop,
        int nblk, double* __restrict__ out);      // [Nch*nblk], nblk == GB (this chunk)
```
1. 🔴 **H2 (blocking): `double s[2K]` must be REGISTERS.** With runtime `ncat`, a channel loop with runtime `(a,b)` is a
   dynamically-indexed local array ⇒ nvcc spills to **local memory** ⇒ `2·Nch·Pn·8` = **3.4 GB local traffic/chunk**
   (~5–10 % of the sweep, scaling as k²) — this, not barriers, is what fails P5. **Fix: template on `K`, fully-unrolled
   `#pragma unroll for(a<2K) for(b=a;b<2K;b++)` with compile-time indices.** K=10 ⇒ 20 doubles ≈ 40 regs (~72 with temps)
   ⇒ 4 blocks/SM, ~40–50 % occupancy at TB=256 on sm_90.
2. **Warp-shuffle reduce, not shared-tree per channel:** 5×`__shfl_down_sync`, lane 0 → `smem[ch*nWarp+warpId]`.
   **One `__syncthreads` per block**, not ~1680. Shared = `Nch·nWarp·8` = 13.4 KB (4 blocks/SM = 54 KB of 228 KB — not
   occupancy-limiting).
3. Epilogue: threads `0..Nch-1` sum their channel's warp partials in **fixed warp order** → `out[ch*nblk+blockIdx.x]`.
4. **Stage-2 device reduce (blue-team):** `Nch·GBmax` D2H is the one real cost — avian nptn≈20 M/nTile=3 ⇒ GBmax≈26 000 ⇒
   **43.7 MB D2H per chunk per sweep** (~0.7 s over 400 sweeps). A second kernel (`Nch` blocks, grid-stride + fixed tree over
   GB) drops it to `Nch·8` ≈ 1.7 KB and removes the host long-double loop over 5.5 M partials. Deterministic.
5. 🔴 **H7 stride trap:** every existing kernel is called with `Pn` bound to the param named `nptn` (:3092/:3118/:3137) ⇒
   buffers are **`Pn`-strided and `Pn` shrinks on the last chunk** (:2805). Index by `Pn`; allocate `Nch*GBmax` but index
   block partials by **`GB`** (this chunk), precedent :2814/:2818.
6. 🔴 **H3 (blocking) `ncat>10 ⇒ hard diagonal fallback`, NOT env-tunable.** `:2640` guards only `ncat>64`;
   `JOLT_FREERATE_HIGHK` can pin `MAXCAT` above the default 10 (`phylotreegpu.cpp:2206-2209`) ⇒ ncat=64 gives `s[128]`
   (1 KB/thread local) and `Nch=8256 > TB=256`, silently breaking the epilogue.
7. 🔴 **H6 Inf/NaN guard:** `Lp=fabs(lh)+pinv*baseinvar` (:482); an underflow at pinv=0 ⇒ `invl=Inf`. In the gradient one Inf
   corrupts one `gradR[c]`; **in the Gram it is squared and corrupts EVERY channel** ⇒ the whole step direction. After D2H:
   `if(!isfinite(H_ij)) → diagonal step this iteration`, and count it.

Determinism end-to-end: no atomics, fixed shuffle/warp/stage-2 order, per-chunk Kahan in the existing `accR` order (:3139-3140),
whole optimizer under a process mutex (:2648).

---

## 4. The step — D-scaled Jacobi spectral solve (SUPERSEDES the LDLᵀ plan)

🔴 **Blue-team SUPERSEDES red's "LDLᵀ + pivot guard": right problem, wrong structure.** We already commit to a Jacobi
eigensolver (§5) on the *same* ≤19×19 matrix; an LDLᵀ *plus* a Jacobi is more code for less capability, and a
pivot-failure fallback silently changes algorithm mid-run (worse for reproducibility than one always-taken path).

**Formulation (dissolves red SEV-3.1 and SEV-3.2):**
1. `D=diag(H_red)`, floored `D_i ← max(D_i, τ·max_j D_j)`, τ=1e-12.
2. `Ĥ = D^{-1/2} H_red D^{-1/2}` — unit diagonal, entries ∈[−1,1]: **the correlation matrix of the scores**, whose
   off-diagonals ARE the degeneracy this whole project is about.
3. Jacobi `Ĥ=VΛ̂Vᵀ` (fixed sweeps, fixed (p,q) order ⇒ bit-reproducible).
4. Step `δ̂ = Σ_i v_i(v_iᵀĝ)/(max(λ̂_i, λ̂_floor)+λ)`, `ĝ=D^{-1/2}g_red`, `δ_red=D^{-1/2}δ̂`.
   No pivots ⇒ nothing to fail; a zero/negative-from-roundoff eigenvalue is handled by the same spectral floor.

**Reduced coords.** θ=(y₀..y_{k-1}, z₀..z_{k-1}); project out ONLY the softmax null `n_z=(0_k;1_k)` — **exact**:
`Σ_c s^z_{p,c}=R_p−(Σ_c bprop[c])R_p=0` per pattern (`softmaxApply` normalizes exactly :3200-3201; the WOPT gate already
asserts `|Σbprop−1|<1e-9`). Do **NOT** project `n_y`: at fixed brlen it is large-curvature (`Σ_c s^y_{p,c}` = the
total-tree-scale score, well-determined) — prior art's "project both" is wrong. `Q=blockdiag(I_k,C)`, `C` = orthonormal
Helmert basis of `{v:Σv=0}`. `H_red=QᵀHQ`, `g_red=Qᵀg`, `δ=Qδ_red`.

**λ floor — DERIVED (blue-team), not asserted.** Measured off job `174154357` artifacts: worst `maxrel` **3.2e-7**, worst
**`maxabs` 1.677e-03** (RGRADRATE ncat=8 pinv=0). 🔴 The floor depends on **`maxabs`, not `maxrel`** (red quoted the wrong
column), and `maxabs` is printed by the shipped gate but **recorded nowhere in the companion — now recorded here**. Note
1.7e-3 is an *upper bound* (the FD's own roundoff floor is 4.7e-6, O(eps²) truncation dominates) ⇒ pin it with a **3-point
eps-ladder (1e-3/1e-4/1e-5) in Phase 1, free.** Derivation: ε_abs ≤1.7e-3; Cauchy–Schwarz `H_cc ≥ g_c²/N` ⇒ `√H_cc ≳330`;
scaled noise `ε̂≈5.1e-6`; trust budget Δ=0.1 in y/z units ⇒ **λ̂_floor = ε̂/Δ ≈ 5e-5 ⇒ use `1e-4` dimensionless on the
D-scaled system**, A/B at 1e-5 and 1e-3. (Red's `1e-7·max diag` has the **wrong reference quantity** — a max-diag floor is
far too permissive on a soft coordinate.)

**Nielsen gain-ratio λ.** `L_pred=0.5·δ_redᵀ(λ·diag·δ_red+g_red)`; `ρ=(lnL_new−lnL_old)/L_pred`; `ρ>0`: `λ*=max(1/3,
1−(2ρ−1)³)`, `ν=2`; else `λ*=ν`, `ν*=2`. Init `λ=τ·max_i diag(H_red)_i` (τ=1e-3). `+δ` **ascends** (H PSD ⇒ `gᵀδ≥0`) and
`L_pred=gᵀδ−0.5δᵀHδ>0` is the correct **maximization** form (both audits confirm the sign).
- 🔴 **H12 accept test:** keep the existing `ln>lnL+1e-9` (:3501) and `dl<tol` (:3523) **unchanged**; use ρ **only** to
  update λ. Minimal seam, preserves convergence semantics.
- 🔴 **H4 clamped steps:** `:3493` clamps `r∈[1e-4,1000]` and `softmaxApply` floors `w` at 1e-4 **and renormalizes** ⇒ the
  applied δ ≠ the solved δ ⇒ ρ compares against a prediction for a different step. **Set a `clamped` bool; skip the ρ-update
  (fall back to ×2/×0.5) on any clamped iteration.**
- 🔴 **H5 floored-weight active set:** after a `w<1e-4` floor+renorm, `w≠softmax(z)` yet `zR=cz` is stored (:3504);
  `∂w_c/∂z_c=0` for a floored class ⇒ its score is wrong and **the Gram squares it**. **Drop floored `z` coords from the
  reduced solve** (precedent: the L-BFGS brlen active-set :3346-3347).
- 🔴 **H9 λ/ν lifetime:** `mu` is declared outside the outer loop (:3293) and persists; λ/ν must likewise **persist across
  outer iterations**, `ν` resetting to 2 on accept.
- 🔴 **H8:** insertion F must honor `JOLT_IR_FREEZE_MODEL` (δ=0), as :3492/:3494 do, or a shipped attribution diagnostic
  dies silently.
- **H10 (blue REFUTES red):** no `JOLT_REM` interaction — `JOLT_REM_EN` is `static constexpr bool = false` (:3182, RETIRED),
  so the EM branch is dead-code-eliminated.

---

## 5. λ̂_min identifiability diagnostic — on a SEPARATE (2k−2) matrix

🔴 **Blue-team RESOLUTION neither red nor v1 proposed: the step matrix and the diagnostic matrix are different objects.**
Step = **(2k−1)** (`n_y` is genuinely informative at fixed brlen). Diagnostic = **(2k−2)**, with **both** gauges projected
out, because identifiability is a question about the **gauge-fixed manifold** — keeping `n_y` inflates `λ̂_max` and biases
the ratio toward "unidentifiable" (red SEV-3.2, confirmed). Two Jacobi calls on ≤19×19 — free.

Report `λ̂_min` and `λ̂_min/λ̂_max` (dimensionless on the D-scaled matrix, ∈[0,2k]). Trust only the **coarse**
identifiable/not call, only once converged (BHHH = true information only at the MLE; the FD-validated gradient means a
1e-6…1e-10 boundary is NOT claimable).

🔴 **whtest C-linkage trap (avoid):** `whtest/eigen_sym.h` exposes `eigen_sym_core` = NR `tred2`+`tqli` in a **C-compiled
GPL** module with header macros `#define NUM_STATE 4`, `#define ZERO 1e-6`, `typedef double DVec20[NUM_STATE]` that would
pollute/clash the CUDA TU. **Do NOT include or link whtest**; `utils/eigendecomposition.cpp` is also off-limits
(non-symmetric rate-matrix path). Ship the standalone cyclic Jacobi.

---

## 6. ~~Lever #2~~ — **DELETED** (both audits, emphatic)

🔴 **The λ_min short-circuit of `phylotesting.cpp:2649` IS approximate ranking — the project's cardinal sin (4/4 real-data
failures).** Read on disk: the loop `for(step<5)` **breaks on the first pass** unless `+R{k}` fit **worse** than nested
`+R{k−1}` (`if (prev_info.logl < new_logl + eps) break;`) — for nested models that is proof of *optimizer failure*. It is a
**rescue** loop, not a routine one. Short-circuiting it ships a known-bad, worse-than-its-own-submodel lnL straight into the
BIC comparison — ranking a candidate on a deliberately under-optimised state, the exact mechanism of the 4/4 failures. The
code itself flags it via `outWarning` (:2665). It is also **largely redundant with Lever A**: if the OPG works, the fit is no
longer worse than `+R{k−1}` and the loop breaks anyway.

✅ **Replacement: the re-seed count becomes a GATE METRIC, not a thing to suppress** — it is a free, direct measure of
optimizer failure. Kill-gate: *"Lever A drives re-seeds to zero."* (This also removes insertions J/K and the `RateFree`
plumbing from the critical path; `out_lambda_min` is retained for reporting only.)

---

## 7. Gate (fail-closed)

- 🔴 **Per-pattern score FD — INDEPENDENT, not a host restatement (red SEV-1.1, blue STRENGTHENS).** v1's "form `s_p` on
  host from §1 and compare to device `H`" is a **self-consistency check**: host and device both implement §1, so a wrong §1
  passes GREEN on both sides — the same false-GREEN class the project keeps getting burned by. **Replacement:** an
  independent per-pattern **finite difference** against the per-pattern log-likelihood. `patlh[ptn]` holds **log L_p**
  (:483; `kj_invl` :804 gives `invl=exp(−patlh)`), and `snapPatlh` (:2966) with the `+pOff`/`Pn` D2H (:3005) is
  **tile-aware by construction** ⇒ nTile>1 works free (do NOT force nTile=1 as the old RGRADCHECK does at :2731). Recipe
  mirrors the passed WOPT/RATE blocks (:3221-3285): snapshot `patlh` at `θ_d±eps`, central-difference → `s_p` directly,
  compare to the formula on K patterns. **eps=1e-4** — and this test is **~1e5× sharper** than the aggregate gate
  (`log L_p`~O(1–10), ulp~1e-15 ⇒ FD floor ~1e-11 abs vs `s_p`~O(0.01–1) ⇒ rel ~1e-9…1e-11), so **rel<1e-8 is achievable
  with room**. Compare against the **sumWN form** (§1 caveat). Cover pinv=0 AND pinv>0, DNA+AA+avian, **nTile>1**.
- **P6 OFF byte-identity (blocking):** `JOLT_OPG` unset ⇒ no kernel, no alloc, :3489-3499/:3518/:3524 untouched ⇒
  bit-identical on DNA+AA+avian+`-p`. Proof-of-build `strings|grep -c kj_opg_gram` (**measured count**, never `grep -q`).
- **P1 crawl:** no candidate at the 401 cap; **re-seed count → 0** (§6).
- **P2/P3 recall+RF (blocking; state-changing ⇒ higher bar):** ON winner == all-tight reference winner AND top-10 BIC order
  on DNA+AA+**real avian**; RF=0 vs the 1e-7 reference over ≥3 seeds. 🔴 **Pre-register P3 PER IDENTIFIABILITY REGIME**
  (red SEV-1.2): claim RF=0 only where `λ̂_min/λ̂_max` is above the floor; below it the deliverable narrows to *detection*.
  Do NOT carry an unconditional RF=0 claim into the dense phase. Also: **RF=0 cannot prove `H` is correct** (any PSD H gives
  an ascent direction and the monotone accept test still converges) — which is exactly why the per-pattern FD is a separate
  blocking gate.
- **P5 wall:** Gram kernel wall as a **measured fraction of the gradient sweep** (target ≪1 %; H2 is the thing that would
  break it). **P7 `-p`:** ON==OFF winner per partition (3 partition-blind gates shipped before — mandatory).
  **P8 λ̂_min:** small where selection is unstable (avian R5–R7), large where stable.
- **Bench BOTH ways:** as fast as tol=1e-2 AND landing on the tol=1e-7 optimum's RF=0/BIC.

---

## 8. REVISED phase order (blue-team: v1's Phase 1 was not a valid kill-gate)

🔴 v1's Phase 1 (diagonal spike, keep `mu`) is invalid for three source-grounded reasons: (a) it **moves state on an
unvalidated H** — v1 put the Gram check in Phase 3, so a wrong per-`c` prefactor makes `H_cc` wrong by that constant
*squared* and the verdict meaningless **in both directions**; (b) the retained **additive** `mu` (init 1.0 :3293) against
`H_cc`~O(1e5–1e7) needs 9–12 quadruplings to matter inside a **14-deep** backtrack (:3480) ⇒ a negative result is
indistinguishable from a damping-scale artifact — **a gate that cannot falsify cleanly is not a gate**; (c) the diagnostic
payload is free and risk-free.

- **Phase 0 — DONE:** GATE-0.5 rate+weight FD (jobs 174153578 / 174154357).
- ✅ **Phase 1 — DONE + PASSED (job `174156911`, binary `cbb9f5a8`, 2026-07-20). H IS VALIDATED.** Templated
  `kj_opg_gram<K>` (compile-time channel indices → registers; warp-shuffle; 1 barrier) + `kj_opg_reduce2` stage-2 +
  self-contained Jacobi. **Check A (independent FD vs `patlh`) PASS on all cells** DNA+AA+avian, pinv=0 AND pinv>0,
  incl. nTile>1 — and the **eps-ladder proves it**: `|analytic−FD|` scales as **eps²** (2.5e-5→2.5e-7→3.8e-9 at
  1e-3/1e-4/1e-5) ⇒ the analytic per-pattern score IS the true `∂logL_p/∂θ` (a redistribution bug leaves a *constant*
  residual). The +I cells at `pinv=0.19` clear at worstabs 1.2e-8 ⇒ the `R_p` normalizer is per-pattern correct. **Check
  B (host-vs-device Gram) PASS everywhere incl. 3-chunk nTile.** `[OPGCOST]` = **0.03–0.14 ms/launch** ⇒ H2
  register-spill did NOT bite (P5 satisfied). Z byte-identical OFF, zero OPG output. RGRADCHECK regressions pass.
  🔴 **1st run (`174156457`) false-FAILED** on the Check-A abs floor (1e-9, ~250× BELOW the eps=1e-4 truncation floor of
  2.5e-7 ⇒ near-zero-score patterns flapped the rel arm — the blue-team's aggregate-flap prediction, now per-pattern);
  **fixed to the hybrid `rel<1e-5 OR abs<1e-5`** (abs floor above truncation, ~1e4–1e5× below any real bug).
  ⭐⭐ **`[OPGLMIN]` WORKS ON THE FIRST RUN AND CONFIRMS THE DEGENERACY HYPOTHESIS DIRECTLY** (§3): λ̂_min/λ̂_max (step
  space) = **0.086–0.11 at R4** (matched to the 4 real classes, well-conditioned) → **3e-7 at R8** (over by 4,
  near-singular) → **9e-14 at avian R6** (real, extreme). The identifiability diagnostic — the novel contribution —
  measures over-parameterisation cleanly. P8 is essentially pre-satisfied by construction; the overnight runs calibrate
  it across the full ladder.
- **Phase 1 (NEW) — build + VALIDATE H; NO step change.** `kj_opg_gram` (§3) + insertions D/E + the independent per-pattern
  FD gate (§7) + λ̂_min **reporting only**. Kill-gates: per-pattern `s_p` rel<1e-8 (pinv=0 **and** >0, DNA+AA+avian,
  **nTile>1**); P6 OFF byte-identity with measured sentinel; the P5 cost micro-benchmark; the 3-point eps-ladder pinning
  ε_abs for the λ floor. **Not state-changing ⇒ the approximate-ranking precedent does not apply.**
- **Phase 2 — diagonal-OPG step WITH multiplicative Marquardt/Nielsen λ** (v1 Phases 1+2 **merged**; `mu` retired on the OPG
  branch). Kill-gate: fixes R4, holds R8/R10/WAG/JTT, iters + reject_frac drop, no `mu`-inflation path, DNA byte-id off.
- **Phase 3 — dense (2k−1), D-scaled Jacobi spectral solve** (§4) + H4/H5/H6/H8/H9 handling. Kill-gate: reaches the 1e-7
  optimum at 1e-2 on DNA+AA+avian+`-p`, P2/P3 per regime, iteration win.
- **Phase 4 — (2k−2) diagnostic matrix + `JOLT_OPG_PROJY` A/B** on the existing `rd_clamps`/`rd_maxm` counters
  (:3191-3192, printed under `JOLT_RDIAG` :3551-3555). Kill-gate: P8 correlation.
- **~~Phase 5~~ — DELETED** (§6).

---

## 9. Risks (revised)

1. **Per-pattern score correctness** — mitigated by the *independent* FD gate (§7), promoted to Phase 1. (The algebra itself
   has now been re-derived by two independent auditors and holds.)
2. **`s[2K]` local-memory spill (H2)** — the real P5 risk; mitigated by templating on K with compile-time channel indices.
3. **Flat direction below gradient accuracy (SEV-1.2)** — the λ̂ floor (1e-4 dimensionless) + P3 pre-registered per regime.
   Honest: where `λ̂_min/λ̂_max` sits below the floor, a full Newton step is **noise-amplifying — less reproducible than the
   damped diagonal it replaces**. The floor *is* the canonicalization (a fixed ridge selects a min-norm-like point
   deterministically).
4. **Buffer timing / stride under tiling (H7)** — launch strictly at insertion E; index by `Pn`/`GB`; explicit nTile>1 cell.
5. **Determinism** — OFF launches nothing and allocates nothing; ON = fixed shuffle/warp/stage-2 order + per-chunk Kahan +
   fixed-sweep Jacobi + self-contained eigensolve (no whtest).

**Graveyard: CLEAR for Lever A** — a conditioned Fisher-scoring Newton step: NOT SQUAREM/Aitken/sequence-accel (3 corpses),
NOT EM redesign (keeps the joint Newton step, replaces only the curvature), NOT approximate/subsample ranking (candidate set,
criterion and comparison untouched). 🔴 **Lever #2 was NOT clear and is deleted (§6).** Framing (red SEV-2.2): drop
"reaches the SAME MLE faster" — on a flat valley that premise is not well-defined. Correct framing: **a state-changing
optimiser improvement that ships ONLY if the new winner is verifiably the correct AND seed-stable MLE.**

---

## 10. Phase-2 GATE — night1 attribution (job `174157175`, canonical `f3f7875f`) ✅ GO

Direct model-arm vs brlen-arm attribution on +R, DNA **and** AA, via `JOLT_IR_FREEZE_MODEL`/`JOLT_IR_FREEZE_BRLEN`
(both flags measured present in the binary, count=1). `model_share = (wallA − wallC)/wallA`, wallC = brlen-only arm.

| dataset | model | wallA | wallC | **model_share** | brlen_share | note |
|---|---|---|---|---|---|---|
| dna | GTR+R5 | 16.57 | 3.78 | **77.2 %** | 6.5 % | A **and** B pin iters=401 |
| dna | GTR+R8 | 18.17 | 3.77 | **79.3 %** | −0.6 % | A **and** B pin iters=401 |
| dna | GTR+G4 | 3.38 | 3.12 | 7.7 % | 9.2 % | control — OPG must NOT touch |
| aa | LG+R5 | 50.84 | 5.51 | 89.2 % | −9.9 % | |
| aa | LG+R8 | 86.30 | 7.98 | 90.8 % | 1.2 % | A **and** B pin iters=401 |
| aa | LG+G4 | 5.83 | 4.83 | 17.2 % | 37.0 % | control |
| avian | GTR+R6 (**real**) | 32.49 | 11.46 | **64.7 %** | 4.5 % | A pins iters=401 |

**Verdict: GO.** The gate criterion was *"if the DNA model-arm share is ~10 %, the OPG is not worth building."* Measured
DNA +R model arm = **77–79 %** (sim), **64.7 %** (real avian) — an order of magnitude over the no-go line, and the first
*direct* measurement of the OPG's upside on the +R DNA target (all prior numbers were inferences or +G/AA/Mode-L transfers).

Corroborations (all on-disk, `attrib.tsv`):
- **+G4 rows (7.7 % DNA / 17.2 % AA) validate the method** — the historical "8–30 %" bound was a **+G figure**, correctly
  low; the OPG's decision to leave α/pinv/Q on diagonal arms is confirmed by measurement, not assumed.
- **The +R model arm is what pins the 401 cap:** every DNA +R cell hits iters=401 in arm A (baseline) **and** arm B (brlen
  frozen) — the model-only arm alone exhausts the cap. That is the exact non-convergence the OPG conditions.
- **End-to-end share (× GAP-1 sweep weighting):** DNA ≈ 0.909 × 0.79 ≈ **72 %** of total MF wall addressable; AA ≈ 0.364 ×
  0.91 ≈ **33 %** (+G candidates dominate the AA sweep and the OPG does not touch them). DNA is the larger end-to-end prize.
- The green-team GAP-1 AA-favoring worry (90.9 % vs 36.4 %) was a *sweep-weight* decomposition; this is a *per-candidate
  wall* decomposition. Both hold; they answer different questions. Per-candidate, the model arm dominates on **both** types.

**Caveat (honest):** freezing an arm changes the trajectory, so shares need not sum to 1 (negatives −0.6 %/−9.9 % are
trajectory noise); these are upper-bound estimates on the OPG's reachable wall, not an exact additive split. Sufficient for
the go/no-go, which they clear decisively.

---

## 11. PHASE 2 DESIGN — the diagonal-OPG step (FIRST state-changing phase) — for red/blue audit

**Greenlit by night1 (§10).** Phase 1 built + validated the empirical-Fisher Gram `H` (default-OFF, byte-identical).
Phase 2 makes `H` *do work*: replace the flaky per-coordinate **secant** curvature of the (y,z) block with the OPG
empirical-Fisher **diagonal**, and retire the additive-`mu` ratchet for a **multiplicative Marquardt/Nielsen λ** trust region.

### 11.1 What is actually broken (source, worktree `iqtree3-opg` `gpu_lnl_intree.cu`)
The +R (y,z) arm steps at `:3776`/`:3778`:
```
ny   = baseY[c] + g_y[c]/(|ddY[c]| + mu);   cr[c]=exp(ny)      // rate arm
cz[c]= baseZ[c] + g_z[c]/(|ddZ[c]| + mu);   softmaxApply       // weight arm
```
`ddY[c]`/`ddZ[c]` are **scalar secant curvatures** (FD of the gradient vs the previous iterate, `:3720-3721`), init `-1e6`
⇒ tiny iter-1 step; `fabs()`'d so sign is discarded. `mu` starts 1.0, **×4 on reject / ×0.5 on accept** (`:3802`/`:3808`).
**Failure mode:** when k>true classes, the near-singular direction lives in the **off-diagonal** of the (y,z) Fisher block;
the diagonal secant cannot see it ⇒ the step there is `g/mu`, over-shoots, is rejected, `mu`×4 ⇒ **all** directions
over-damp ⇒ crawl ⇒ 401 cap (night1: every DNA/AA +R cell pins iters=401). Phase 2 fixes the **diagonal** curvature +
the damping law; the **off-diagonal** win is Phase 3 (dense solve) — so the gate asks Phase 2 to *fix R4* and *hold* R8/R10.

🔴 **PRECEDENT (WORK-LOG): a prior `curvFloor` curvature-flooring attempt was REVERTED after regressing DNA +R8/R10.**
This is the exact trap. The gate MUST prove R8/R10/WAG/JTT do not regress (winner + lnL within noise of canonical).

### 11.2 The step (ON-path = `opgOK && g_opg_step`; OFF = canonical, byte-identical)
Extract the OPG Fisher **diagonal** from the packed upper-triangle Gram `opgH` (Phase-1 layout, `kj_opg_gram`): channel of
`(a,a)` is `diagIdx(a) = a*N2 − a*(a−1)/2`, `N2=2*ncat`. So per category `c`:
`H_yy[c] = opgH[diagIdx(c)]`, `H_zz[c] = opgH[diagIdx(ncat+c)]` — both `= Σ_p f_p·s²  ≥ 0` (PSD by construction).
The score identity `g_θ[c] = Σ_p f_p·s^θ_{p,c}` (validated Phase 1 Check A) makes `H` the empirical Fisher for the SAME
gradient `g_y=catRate·gradR`, `g_z=gzR` ⇒ `H⁻¹g` is dimensionally a Fisher-scoring step.

**Unified multiplicative-LM over the whole joint trial (mu retired on the OPG branch):** every arm `i` steps
`Δ_i = g_i / ((1+λ)·C_i + ε_i)` with curvature `C_i` = { (y,z): `H_yy`/`H_zz`; brlen: `|g_ddf[e]|`; α: `|ddA|`;
pinv: `|ddP|`; Q: `|ddQ[k]|` } — i.e. **the non-(y,z) arms keep their existing curvature estimates**, only their damping
law changes from additive `mu` to multiplicative `(1+λ)`. `ε_i` = tiny floor (brlen `1e-9` to match the old `mu` floor;
(y,z) `1e-30`, guards a dead category where `g` and `H` are both ~0).

**Acceptance UNCHANGED:** accept iff `ln > lnL + 1e-9` (identical convergence semantics; `dl<tol ⇒ conv`). λ only governs
the **damping update**:
- predicted increase `P = Σ_i Δ_i·(g_i − ½·C_i·Δ_i)` (exact for the per-arm quadratic model; each term ≥0 ⇒ `P≥0`).
- gain ratio `ρ = (ln − lnL)/P` (guard `P>1e-300`; if `P≈0` the block is converged).
- **accept:** `λ ← λ·max(1/3, 1−(2ρ−1)³)`, clamp `[λmin,λmax]`, `ν←2`.  **reject:** `λ ← λ·ν`, `ν←2ν` (Nielsen 1999).
- `λ0=1.0` (start conservative ≈ old `mu=1.0`), `λmin=1e-7`, `λmax=1e7`; all env-tunable
  (`JOLT_OPG_LAMBDA0/LMIN/LMAX`) for the gate sweep.

### 11.3 Known property (state it, don't let red-team "discover" it)
OPG (BHHH) `H=Σf_p s_p²` = observed Fisher **only at the MLE**; far from it `Σf_p s_p² = N·(Var+mean²)` **over-estimates**
curvature ⇒ steps are **more conservative** (smaller, safe, never overshoot) but **slower** far out. The Nielsen λ trust
region is exactly the standard mitigation (good model ⇒ ρ≈1 ⇒ λ shrinks ⇒ bigger steps). This is the doc's pre-listed
"BHHH poor far from MLE" risk, handled by construction.

### 11.4 Flags / scope / determinism
- New: `g_opg_step` ← `JOLT_OPG_STEP` (implies `g_opg_on`; needs the Gram). Phase-1 diagnostics (`JOLT_OPG`,
  `_GRAMCHECK`, `_LMIN`) keep gram-only (no step). **OFF (neither set) ⇒ the `|dd|+mu` path runs verbatim ⇒ byte-identical.**
- Scope: `freeRate==1` only, `ncat∈[2,10]` (else `opgNCH=0 ⇒ opgOK=false ⇒` diagonal-mu fallback). No change to +G/+I-only/Q-only.
- Determinism: `opgH` is the deterministic Phase-1 reduction; the step is host FP64 arithmetic ⇒ bit-reproducible.

### 11.5 Kill-gate (state-changing ⇒ RECALL-grade)
1. **Byte-identity OFF** — `JOLT_OPG_STEP` unset ⇒ lnL+params bit-identical to canonical `f3f7875f`, DNA **and** AA
   (sentinel `strings|grep -c` a new marker; md5≠canonical; OFF lnL == canonical lnL to the ULP).
2. **fixes R4** — matched-k +R4 (well-conditioned, λ̂_min/λ̂_max~0.1): iters < 401 cap, `dl<tol` reached, lnL ≥ canonical−1e-6.
3. **holds R8/R10/WAG/JTT** (the `curvFloor` trap): winner + lnL within noise of canonical, no crash/NaN, DNA **and** AA.
4. **iters + reject_frac ↓** — `[JOLT]` census iters and `nRej/nLnLEval` both drop vs canonical on the +R cells
   (proof-of-effect: ON must differ from OFF, else the step is a no-op).
5. **no mu-inflation** — `JOLT_IR_CONVTRACE` shows λ (not mu) adapting; λ does not pin at λmax; canonical mu→1e9 gone.
6. **regressions** — `[OPGLMIN]` still fires, `[OPGCOST]` still negligible (Phase-1 carry-over); real **avian** GTR+R6 cell.

### 11.6 Primary risks for red-team (ranked)
- **R-A (top):** the brlen/α/pinv/Q arms' damping law changes (additive `mu` → multiplicative `(1+λ)`). Even though their
  curvature `C_i` is unchanged, iter-1 damping differs (old `mu=1.0` additive vs `(1+λ0)|dd|`) ⇒ could destabilise a
  well-behaved arm. **Fallback if the gate regresses: keep brlen/α/pinv/Q on the verbatim `|dd|+mu` path and put ONLY the
  (y,z) block on OPG+Nielsen**, with ρ computed over the (y,z) sub-block only (accept the mild gain-ratio contamination, or
  add one (y,z)-only evalLnL for a clean ρ). This is the minimal-perturbation variant; unified is primary because it gives
  a clean whole-step gain ratio and genuinely retires mu.
- **R-B:** gain-ratio contamination in the unified form is ZERO (P sums all arms) — but if an arm's `C_i` is a poor
  curvature (secant noise), its predicted term mis-estimates and biases ρ. Mitigation: ρ only *modulates* λ; acceptance is
  the exact lnL test, so a biased ρ slows adaptation but cannot accept a bad step.
- **R-C:** `H_yy/H_zz` read staleness — must be the CURRENT base-state Gram. `opgH` is filled by `computeGradient` each
  outer iter (Phase-1 per-sweep reset `:3156`); extract `H_yy/H_zz` in the same block as `g_y/g_z` (`:3718`), before the
  backtrack loop. Assert `opgH` non-zero when opgOK (a zero Gram ⇒ silent fallback bug).
- **R-D:** the softmax gauge — `s^z` already has the `n_z` null projected by construction (`s^z_{p,c}=…−bprop[c]·R_p`),
  so `H_zz` is the correct reduced-space diagonal; no extra gauge handling needed at the diagonal level (the null only bites
  the DENSE solve in Phase 3).

---

## 12. Phase 2 AUDIT LOG (red done, blue in flight) — fold into a v2 §11 before build

**Phase-1 tree COMMITTED** `1bb82e14` (on `a6fc4d39`) — SEV-3.4 closed; Phase-2 code layers on top, bisectable. Working
tree was dirty; the validated Phase-1 Gram is now pinned to the bytes (job `174156911`, binary `cbb9f5a8`).

**RED-TEAM = REVISE-FIRST (no hard blocker).** Verified SOLID vs source: diagIdx `a·N2−a(a−1)/2` + y/z channel map;
score scaling exact (`g_y=catRate·gradR`, kernel `cRcP=catRate·catProp_v` — both carry catProp_v ⇒ H is the Fisher for the
SAME g_y); gain-ratio `P≥0` + Nielsen factor ∈[1/3,2]; `opgH` fresh at extraction (computeGradient at loop top
resets+accumulates); softmax null handled by shift-invariance at the diagonal level; byte-id OFF plan-sound; `mu` at
`:3577` is the persistent home for λ/ν; exact-accept (`:3785`) is the safety net (a wrong H ⇒ slower/different ascent,
never a non-ascent); curvFloor precedent REAL (κ·rN floor fixed AA-R4 but regressed LG+I+R8 −0.41 / R10 −2.6, gate
`173910972`). **Must-fix (ordered):**
1. **SEV-2.1** — preserve the `JOLT_IR_FREEZE_MODEL?0.0:` guard in the rewritten (y,z) step (else §10's attribution gate
   silently breaks, uncaught by §11.5); declare λ/ν at `:3577` beside `mu` (NOT in the outer loop ⇒ else reset each iter ⇒
   no adaptation ⇒ fails "fix R4"); `ν←2` on accept.
2. **SEV-2.2/2.4** — `ε=1e-30` for (y,z) is UNSAFE (near-dead category: H_yy≈0, g_y small-nonzero ⇒ Δ≈1e10, clamped to
   [1e-4,1000] masking the blow-up, poisoning ρ; old `mu≥1e-9` was safe). Fix: relative floor `ε←max(1e-30, τ·max_c(H_yy,
   H_zz))`, τ=1e-12 (mirrors §4's D-floor) OR active-set drop of near-zero-H coords. Plus **clamp-aware ρ**: skip the
   Nielsen update on clamped-rate / floored-weight iterations (applied Δ ≠ solved Δ ⇒ contaminates ρ exactly in R8/R10).
3. **SEV-2.3** — flip R-A: make the **(y,z)-only variant PRIMARY** (brlen/α/pinv/Q stay verbatim on `|dd|+mu`); the unified
   single-λ multiplies EVERY C_i so a joint reject over-damps all arms = the renamed pathology; brlen measured small/benign
   (§10). Or run both as gate arms and let R8/R10 decide.
4. **SEV-3.x** — reconcile §11.2 `P=ΣΔ(g−½CΔ)`/λ0=1.0 vs §4 `L_pred=½δ(λ·diag·δ+g)`/λ0=τ·maxdiag; propagate the pinv=0
   `rN` vs `sumWN` ~3.7e-5 caveat; reconcile doc line-refs (some point at the rgradrate tree, not the opg build tree).

**BLUE-TEAM: in flight** (optimisation/implementation of the above — clean ρ for (y,z)-only, active-set vs relative-ε
cost, λ/ν state machine, minimal gate matrix). Fold red+blue into a **v2 §11** before the gate-first build.

---

## 13. PHASE 2 — v2 BUILD SPEC (red+blue folded; THIS is what gets built) ✅

**BLUE-TEAM = REVISE-FIRST (no blocker).** Re-verified the red-team SOLID list on disk; revised the ρ recipe + the
floor; added two guards; confirmed (y,z)-only primary + single-switch build. Consolidated, authoritative build list:

**Variant: (y,z)-ONLY primary.** brlen/α/pinv/Q stay VERBATIM on `|dd|+mu` (`:3765-3769`) with mu's ×½/×4
(`:3802`/`:3808`). Only `:3776`/`:3778` change. Rationale: identical GPU cost to unified (one evalLnL/backtrack either
way), minimal diff, leaves four measured-benign arms byte-identical; unified's sole edge ("one knob") evaporates once ρ is
computed whole-step (below), while its R-A regression risk on proven arms remains. Unified kept as `JOLT_OPG_STEP=1`
confirmatory only (one cell).

1. **Flags/wire.** `g_opg_step` ← `JOLT_OPG_STEP` (int: 0 off / 2 (y,z)-only primary / 1 unified confirm), forces
   `g_opg_on=true` (extend the `g_opg_init` block; Gram allocs under `opgOK` `:2958`). Step gated `opgOK && g_opg_step`.
   `opgOK==false` (ncat∉[2,10], or the per-coord Inf guard below) ⇒ verbatim diagonal-mu fallback. **Preserve the
   `JOLT_IR_FREEZE_MODEL?0.0:` guard** in the rewritten `:3776`/`:3778` (else §10's attribution gate dies silently, uncaught).
   **Keep the secant machinery** (`:3719-3722`) alive for the fallback.
2. **λ/ν state machine.** Declare `double lam=1.0, nu=2.0;` beside `mu` at **`:3577`** (persistent across outer iters — NOT
   inside `for it`, else no adaptation ⇒ fails "fix R4"). Clamp `lam∈[1e-7,1e7]`; env `JOLT_OPG_LAMBDA0/LAMMIN/LAMMAX`
   (LAMMIN/LAMMAX, NOT LMIN/LMAX — the latter is the λ_min diagnostic flag).
3. **(y,z) step** (`:3776`/`:3778`): `Δθ_c = g_θ[c] / ((1+lam)·H_θθ[c] + ε)`, `H_yy[c]=opgH[diagIdx(c)]`,
   `H_zz[c]=opgH[diagIdx(ncat+c)]`, `diagIdx(a)=a·2ncat − a(a−1)/2`. Extract H once per outer iter at `:3718` (opgH fresh).
4. **Relative ε floor (continuous, deterministic — NOT a threshold-drop).** `ε = max(1e-30, 1e-12·max_c{H_yy,H_zz})`
   (mirrors §4/§5 D-floor τ=1e-12). A magnitude-drop active-set is a seed/build-flip hazard — forbidden.
5. **Whole-step ρ (option d).** `P = Σ_i Δ_i(g_i − ½·C_i·Δ_i)` over ALL stepping arms (brlen `C=|g_ddf|`, α `|ddA|`,
   pinv `|ddP|`, Q `|ddQ|`, y `H_yy`, z `H_zz`), applied Δ, fixed summation order (edges→α→pinv→Q→y→z; determinism).
   Each term ≥0. `ρ = (ln−lnL)/P` (guard `P>1e-300`). Zero extra cost; no brlen contamination (dominates option a).
6. **Nielsen λ update + clamp-aware ρ.** accept & !clamped: `lam·=max(1/3, 1−(2ρ−1)³); nu=2`. accept & clamped:
   `lam·=0.5; nu=2`. reject (any): `lam·=nu; nu·=2`. mu keeps ×½/×4 untouched. `clamped` = any rate/α/pinv/Q/brlen clamp
   OR any weight floored to `1e-4` (`:3392`); floored-z coords excluded from BOTH the step and P (exact `w==1e-4` event,
   H5 — not a magnitude threshold).
7. **Step-path Inf/NaN guard (blue add).** Before dividing, per coord require `isfinite(H_yy[c] & H_zz[c] & g_y[c] &
   g_z[c])`; else that coord falls back to `|dd|+mu` and is counted. (The Gram SQUARES scores ⇒ one underflow pattern ⇒
   H=Inf.)
8. **Diagnostics.** Extend `[IRCONV]` (`:3810`): print `lam, nu, rho, max|g_y|, max|g_z|` (distinguishes genuine conv
   grad≈0 from λ-stall grad≠0/λ-high — the renamed mu→1e9 false-conv trap). Keep `[OPGCOST]`/`[OPGLMIN]`.
9. **λ0 note (not a "reconcile"):** §13 `λ0=1.0` is MULTIPLICATIVE on raw H (Phase-2 diagonal, `Δ=g/2H` at iter-1);
   §4 `λ0=τ·maxdiag` is ADDITIVE on the D-scaled unit matrix (Phase-3 dense). Different solvers — state, don't merge.

### 13.1 Gate (single env-switch build; one job)
| purpose | cells | pass |
|---|---|---|
| **OFF byte-id** (block) | DNA GTR+R5, AA LG+R5, avian GTR+R6, +1 `-p` partitioned | md5 OFF == canonical `f3f7875f`; `strings\|grep -c JOLT_OPG_STEP`≥1; OFF lnL==canonical |
| **fix R4** | DNA GTR+R4, AA LG+R4 | iters<401, `dl<tol`, lnL ≥ canonical−1e-6 |
| **hold (curvFloor trap)** | DNA GTR+R8/R10, AA LG+R8/R10, WAG+R8, JTT+R8 | winner+lnL within noise of canonical |
| **proof-of-effect** | R8 cells | ON iters AND nRej/nLnLEval both drop vs OFF |
| **λ-adapt / no-inflation** | `[IRCONV]` | lam adapts, NOT pinned λmax; **mu also NOT at 1e9** (check both) |
| **carryover** | `[OPGLMIN]`/`[OPGCOST]` fire; real avian GTR+R6 | present + negligible |
| **confirm** | unified `=1` on DNA GTR+R8 only | not decisively better than (y,z)-only |

---

## 14. Phase 2 IMPLEMENTED + gate submitted (job `174158000`, 2026-07-20)

Coded in worktree `iqtree3-opg` on top of the committed Phase-1 `1bb82e14` (UNCOMMITTED until the gate passes; the build
job builds from the working tree). Edits to `gpu_lnl_intree.cu` (all (y,z)-only, mu untouched for the other arms):
- flags: `g_opg_step`←`JOLT_OPG_STEP` (nonzero=on, forces `g_opg_on`); `g_opg_lam0/lmn/lmx`←`JOLT_OPG_LAMBDA0/LAMMIN/LAMMAX`.
- `opgStepOn = freeRate==1 && opgOK && g_opg_step`; `lam/nu/opgFallN` PERSISTENT beside `mu`.
- H-diagonal extraction each outer iter (diagIdx) + relative floor `opgEps=max(1e-30,1e-12·maxH)` + per-coord finite guard
  `opgSC[c]`.
- (y,z) step: `dyc=(opgSC[c]?(1+lam)·Hyy+opgEps:|ddY|+mu)` — OFF ⇒ `opgSC.size()==0` ⇒ fallback ⇒ bit-identical. FREEZE_MODEL
  guard preserved.
- whole-step `Ppred` (option d) over all arms + `clmp` (any clamp/floor). accept: `!clmp`⇒Nielsen `lam·=max(1/3,1−(2ρ−1)³)`
  else `lam·=0.5`, `nu=2`; reject: `lam·=nu, nu·=2`. mu keeps ×½/×4.
- Inf-guard (opgSC), `[IRCONV]` extended (lam/nu/maxgy/maxgz), `[OPGSTEP]` engagement summary.

Gate `gems_opg_phase2.sh` (job `174158000`, ~4 h): clean build + B/P (md5≠canon, `strings|grep -c JOLT_OPG_STEP`, canon
lacks it) + Z (OFF==canon on DNA/AA/avian **+ partitioned `-p`**) + R4 fix + HOLD R8/R10/WAG/JTT (hard-fail >0.5 nat
regression, ⚠️ inspect 0.05–0.5 — curvFloor-R8 was −0.41) + EFF (iters+nRej drop) + LAM (lam adapts, not pinned) + CO
(OPGLMIN/OPGCOST carryover). Script scanned: bash -n clean, grep -c (not -q), JOLT_DIAG=1 for the census, .console capture.
**Awaiting result.**

---

## 15. Phase 2 GATE RESULT — 🔴 FAIL (job `174158208`, binary `74b63884`, 2026-07-20). Honest negative.

Ran the full §13.1 matrix. **PASSED:** B/P; **Z byte-id OFF == canonical on DNA GTR+R5 / AA LG+R5 / avian GTR+R6 / a
partitioned `-p` cell (all exact)** — the ship-safety invariant is solid; CO ([OPGLMIN] avian ratio 1.7e-6, [OPGCOST]
0.077 ms). **FAILED:**
- **avian GTR+R6: ON −11217049 vs OFF −11216886 = −163 nats (HARD FAIL), lam pinned at ceiling 1e7.**
- **EFF: NO work-drop on any degenerate cell** — R8/R10/avian still 401-capped, nRej *higher* on ON (e.g. dna_r8 225 vs 206).
- aa_r4 FIX: iters 48 > 44 (didn't drop). DNA r4 FIX passed (39<48, identical lnL). 3 HOLD cells 0.17–0.38 nat worse (⚠️,
  under the 0.5 hard-fail): dna_r8 −0.31, aa_r8 −0.17, aa_r10 −0.38. wag_r8 **+3.6**, jtt_r8 **+11.9**, dna_r10 +0.07 (ON better).

**ROOT CAUSE (mechanistic, not a bug — sign/extraction verified correct by DNA-R4 converging to the identical lnL):** the
**diagonal empirical Fisher is COUPLING-BLIND.** On the degenerate cells — where the pathology is *off-diagonal* (two
categories collapse) — the per-coordinate diagonal step is a worse direction than the secant, so the joint trial is
rejected → Nielsen ratchets `lam` to the ceiling → `(1+lam)·H` with H~1e6 makes the (y,z) step ≈0 → **the (y,z) arm FREEZES
at the warm-seed** → −163 nats on avian (the most degenerate cell, OPGLMIN ratio 9e-14). The worse the coupling, the worse
the diagonal does (avian ≫ R10 ≫ R8). **⇒ DIRECT EMPIRICAL PROOF that the off-diagonal / DENSE solve (Phase 3) is the
necessary mechanism.** The diagonal was never going to fix an off-diagonal degeneracy; the gate proved it, and worse, showed
it can REGRESS without a floor.

**TWO fixes:**
1. **MANDATORY SAFETY (Fix A), needed by EVERY OPG phase:** when the OPG step saturates `lam` (or after N (y,z)-rejects),
   **fall back to the secant `|dd|+mu` step for the (y,z) arm** so it never freezes — bounds the downside to baseline
   (guarantees HOLD; recovers avian). Env `JOLT_OPG_LSAT` (fallback when `lam ≥ LSAT`, default 1e3). Keep P/ρ consistent
   with the actually-taken step.
2. **THE REAL WIN = Phase 3 (dense (2k−1) D-scaled solve)** — captures the coupling the diagonal is blind to. Phase 2's
   value was only ever to de-risk the Nielsen machinery + this fallback; that is now done. **Reconsider shipping a
   standalone diagonal Phase 2 at all** — with Fix A it is *safe but buys ~nothing on the target* (marginal DNA-R4 only).
   Likely fold Fix A into Phase 3 rather than ship the diagonal.

**Phase-2 source NOT committed** (gate failed). tol-ladder deferred (a regressing binary would show uniform regression).

---

## 16. PHASE 3 — dense (2k−1) solve + Fix A (user-chosen path after §15). Design for red/blue audit.

**Why:** §15 proved the diagonal is coupling-blind — it CAN'T fix the off-diagonal degeneracy and freezes without a floor.
Phase 3 replaces the per-coordinate (y,z) step with the DENSE spectral solve of §4 (captures the coupling), and bakes in
**Fix A** (no-regression fallback). Builds on the Phase-2 worktree (`iqtree3-opg`, binary `74b63884`, UNCOMMITTED). REUSES
from Phase 2: the full Gram `opgH` (all NCH channels, not just the diagonal), the persistent Nielsen λ/ν state, whole-step
ρ (option d), clamp-aware skip, FREEZE_MODEL guard, byte-id-OFF gating, `[IRCONV]`/`[OPGSTEP]`. REUSES from Phase 1:
`jolt_jacobi_eig` (must be EXTENDED to return eigenVECTORS — see N1).

### 16.1 The dense (y,z) step (replaces the Phase-2 per-coord `dyc/dzc` loop), computed ONCE per backtrack
Per §4, with the current `lam`:
1. **Assemble** full symmetric `H` (2k×2k) from `opgH`'s upper-triangle (channel `(a,b)` → `H[a][b]=H[b][a]`).
2. **Active set** (H5): drop z-coords whose BASE weight is at the floor (`bprop[c] ≤ 1e-4·(1+1e-9)`) — known upfront from
   the base state, so the reduced dimension is fixed for the whole backtrack loop of this outer iter. Active dims =
   k (all y) + m (non-floored z), m≤k.
3. **Reduce** (§4): `Q=blockdiag(I_k, C)`, `C`= orthonormal Helmert basis of `{v∈R^m : Σv=0}` (m−1 cols) — projects out
   ONLY the softmax null `n_z` over the ACTIVE z-coords. `H_red=QᵀHQ` (dim k+(m−1)=`nr`), `g_red=Qᵀg`, g=(g_y;g_z_active).
4. **D-scale + spectral solve** (§4 verbatim): `D=diag(H_red)` floored `max(D_i,1e-12·maxD)`; `Ĥ=D^{-1/2}H_red D^{-1/2}`;
   Jacobi `Ĥ=VΛ̂Vᵀ`; `δ̂=V·[v_iᵀĝ/(max(λ̂_i,λ̂_floor)+lam)]`, `ĝ=D^{-1/2}g_red`; `δ_red=D^{-1/2}δ̂`. `λ̂_floor=1e-4` dimensionless.
5. **Map back**: `δ=Qδ_red` → `δ_y`(k), `δ_z_active`(m); floored z-coords get `δ_z=0`. Apply exactly as Phase 2 (FREEZE_MODEL
   ⇒ δ=0; rate clamp [1e-4,1000]; softmaxApply).
6. **Block predicted increase** (for whole-step ρ, option d): `P_yz = g_red·δ_red − ½·δ_redᵀH_red δ_red`; add the
   non-(y,z) arms' diagonal terms as in Phase 2. `P_yz≥0` (H_red PSD).

### 16.2 Fix A — mandatory no-regression fallback (the §15 lesson)
The (y,z) arm falls back to the Phase-2/canonical `|dd|+mu` per-coord step for THIS backtrack when ANY of:
- `lam ≥ JOLT_OPG_LSAT` (default 1e3) — the trust region has collapsed (the §15 freeze trigger);
- the dense solve produced a non-finite δ (Inf/NaN in H, g, or a solve underflow);
- active set too small (`nr < 2`, e.g. k=2 with a floored z) — nothing to solve densely.
On fallback the ρ still updates λ from the ACTUAL (baseline) step, so a good baseline accept shrinks λ and the dense solve
re-engages next iteration (self-recovery). **Guarantees HOLD:** the OPG can never do worse than the `|dd|+mu` baseline.
Because the dense direction captures the coupling, the EXPECTATION is that the fallback rarely fires on the degenerate
cells (unlike the diagonal, which triggered it constantly) — that is the Phase-3 hypothesis the gate tests.

### 16.3 New code (deltas from Phase 2)
- **N1:** extend `jolt_jacobi_eig(a,n,ev)` → `jolt_jacobi_eig(a,n,ev,V)` accumulating the rotation product into `V` (n×n).
  Phase-1's `[OPGLMIN]` callers pass `V=nullptr` (eigenvalues only) — keep that overload/guard so Phase-1 diagnostics are
  unchanged.
- **N2:** host helper `jolt_opg_dense_step(H2k, g_y, g_z, activeZ, lam, out δ_y, δ_z, out P_yz, out ok)` — assemble/reduce/
  D-scale/Jacobi/solve/map-back. All host FP64, deterministic (fixed Jacobi sweeps + fixed reduce order). ≤19×19.
- **N3:** replace the Phase-2 per-coord `dyc/dzc` block with: compute δ once via N2 (if dense-active), else the `|dd|+mu`
  fallback per coord; stage cr/cz from δ; accumulate P_yz. Keep the Nielsen update, clamp-aware skip, FREEZE_MODEL, Inf
  guard, `[IRCONV]`/`[OPGSTEP]` from Phase 2 verbatim.
- **N4:** the Helmert basis `C` (m×(m−1)) — standard closed form, built once per active-set size; deterministic.

### 16.4 Gate (same §13.1 matrix + the decisive new asserts)
Byte-id-OFF (already PERFECT — must stay) + fix-R4 + **HOLD R8/R10/WAG/JTT with NO −0.05 regression** (Fix A must make this
airtight) + **avian GTR+R6 must NOT regress** (the §15 −163 nat failure must be gone) + **EFF: iters/nRej DROP on the
degenerate cells** (the real win the diagonal couldn't deliver) + `[IRCONV]` λ NOT pinned + a per-backtrack fallback counter
(`[OPGSTEP]` extended: how often Fix A fired — if it fires every iteration on avian, the dense direction is still bad and we
learn that honestly). CO carryover. Then, only if it HOLDS: the ON tol-ladder (1e-7→1e-2) = the payoff test.

### 16.5 Open risks for the audit
- Does assembling the full 2k×2k `H` from `opgH` per backtrack cost too much? (2k≤20, ≤19×19 Jacobi × ≤14 backtracks ×
  iters — host µs; confirm negligible vs evalLnL.)
- The active-set (base-floored z) changes the reduced dim — is the Helmert basis rebuild per outer-iter correct and
  deterministic? Does a mid-run active-set change break the λ/ν trust-region continuity?
- Does the dense δ actually get ACCEPTED more than the diagonal on avian (the whole premise)? Only the gate answers this;
  if Fix A fires constantly, Phase 3 also fails and the honest conclusion is "the +R degeneracy is not fixable by a
  Fisher-scoring step at all" (→ the λ_min DETECTION deliverable, per the Nguyen-2018 bimodality risk).

---

## 17. Phase 3 AUDIT LOG (red done, blue in flight) — fold into a v2 §16 before build

**RED-TEAM = REVISE-FIRST (no blocker).** ⭐ **KEY: the assemble→reduce→D-scale→Jacobi pipeline ALREADY EXISTS + is
Phase-1-validated** in `[OPGLMIN]` (`:3542-3574`, full case): H-assembly `:3548`, Helmert `:3550-3553`, Q reduce `:3557-3559`,
H_red=QᵀHQ `:3560-3562`, D-floor+correlation `:3563-3566`, Jacobi `:3568`. Phase 3 ADDS only: eigenvector-based solve
(δ̂=Σv_i(v_iᵀĝ)/(max(λ̂_i,floor)+lam)), the active-set, Fix A. Verified SOLID: full-H coupling correct (H_yz cross-block is a
real channel, reconstructed at :3548); s^z sums to 0 ⇒ n_z exact null (full case); NOT projecting n_y correct; eigenvector
Jacobi deterministic + δ̂ sign/order-invariant, one caller (:3568), nullptr-V keeps OPGLMIN byte-id; P_yz≥0 proven; byte-id-OFF
holds; cost negligible.
**Must-fix (ordered):**
1. **SEV-1 — "Fix A guarantees HOLD" is FALSE.** It only neutralizes the specific lam-saturation FREEZE (likely recovers
   avian −163). Two regressions survive with lam<LSAT (Fix A dormant): (a) `dl<tol=1e-7` firing on a small accepted step at
   LARGE gradient (lam-stall false-conv — the [IRCONV] trace DETECTS but doesn't PREVENT); (b) flat-valley worse-optimum.
   Fix: downgrade claim; harden `conv` = `dl<tol AND small maxgy/maxgz`; the §16.4 −0.05 HOLD gate is the real safety net;
   optionally retry the baseline (y,z) step within the SAME backtrack on dense-reject (ON tightly dominates baseline).
2. **SEV-2a — active-set Helmert loses the EXACT null when m<k** (a z floored): Σ_active s^z=−Σ_floored s^z≠0 ⇒ projecting
   all-ones-over-active removes a real DOF, on exactly the degenerate cells. Fix: don't Helmert when m<k (D-floor+λ̂-floor
   regularize) OR bound the residual OR re-derive. Drop "exact." **(Blue is assessing: drop the active-set entirely for
   full-Helmert-always + reg — would dissolve 2a/2c/2d.)**
3. **SEV-2d — membership knife-edge:** floored bprop=1e-4/tot sits ON the boundary ⇒ FP-flip nondeterminism. Margin ≥2e-4;
   reconcile base(bprop) vs trial(cw) floor tests.
4. **SEV-2c — mid-run active-set change breaks λ/ν continuity:** reset ν=2, nudge lam→lam0 on change.
5. **SEV-2b — floor coupling:** after good accepts lam→lmn=1e-7, so on a near-null the denom collapses to λ̂_floor=1e-4 ⇒
   ~1e4× amplification ⇒ big δ ⇒ clamps. Fix: raise g_opg_lmn ≳ λ̂_floor; add dense-clamp-rate to the gate.
6. **SEV-3:** Phase-3-UNIQUE proof-of-build sentinel (`JOLT_OPG_STEP` already ships in `74b63884`! use `JOLT_OPG_LSAT` or a
   dense symbol); `{dense_tried, dense_accepted, fixA_fired}` triplet split by trigger; oscillation HYSTERESIS (latch
   baseline until lam<LSAT/10 — else nRej rises and fails EFF); HOIST the eigendecomp out of the backtrack (base-fixed;
   only lam varies in the solve); FREEZE_MODEL short-circuit; P_yz on Fix-A backtracks from the ACTUAL baseline step.

**BLUE-TEAM: in flight** (active-set-vs-drop, refactor-vs-duplicate the pipeline, Fix-A hardening combo, floor coupling,
determinism, gate delta). Fold red+blue into a **v2 §16** before build.

---

## 18. PHASE 3 — v2 BUILD SPEC (red+blue folded; THIS is what gets built) ✅

**BLUE = REVISE-FIRST.** Headline: **DROP the active-set → FULL-HELMERT-ALWAYS** (fixed `nr=2k−1`). The floored class's
`s^z=O(1e-4)` is intrinsically tiny and already neutralized (push-up unfloors it = desirable; push-down re-floors ⇒
`clmp` ⇒ ρ-skip ⇒ no-op under the monotone `ln>lnL+1e-9` accept), so the active-set solved a non-problem while creating
SEV-2a/2c/2d. Full-Helmert = the EXACT `[OPGLMIN]` `report("step2k-1",false)` reduction (Phase-1-validated). Ordered build:

1. **Full-Helmert-always** (nr=2k−1, fixed every outer iter). No membership test, no variable dim, no λ/ν discontinuity,
   no knife-edge. Fix-A's `nr<2` trigger gone (nr≥3 for k≥2).
2. **Extend `jolt_jacobi_eig(a,n,ev,V=nullptr)`** — accumulate the rotation product into `V` (n×n) in the SAME rotation loop
   (`:923-924`); `V=nullptr` ⇒ eigenvalues-only ⇒ OPGLMIN byte-identical (its lone caller `:3568`).
3. **Share the reduction:** extract the OPGLMIN `report` body (`:3557-3568`) into `jolt_opg_reduce(Hf,N2,projY) →
   {nr,Q,Hr_corr,D,ev,V?}`. OPGLMIN calls it twice (`projY=false/true`, `V=nullptr`) — **assert its `lmin/lmax/ratio`
   print UNCHANGED to full precision vs `cbb9f5a8`** (converts refactor risk into a checked invariant). Phase-3 calls once
   (`projY=false`, V on).
4. **Split + hoist:** `jolt_opg_reduce` runs ONCE per outer iter (inputs `opgH`/`g_y`/`g_z`/base-`bprop` are base-fixed;
   only `lam` varies per backtrack). Per backtrack: `jolt_opg_solve`: `δ̂=Σ_i v_i·a_i/(max(λ̂_i,λ̂_floor)+lam)` (`a_i=v_iᵀĝ`,
   `ĝ=D^{-1/2}g_red` precomputed) → `δ_red=D^{-1/2}δ̂` → `δ=Qδ_red` → `δ_y`(k), `δ_z`(k) → `P_yz=g_red·δ_red−½δ_redᵀH_red δ_red`.
   Result-neutral (base-fixed ⇒ bit-identical to per-backtrack recompute).
5. **Dense-phase constants (do NOT inherit Phase-2's):** `lam` is ADDITIVE on the D-scaled correlation matrix ⇒ lam-floor
   `= λ̂_floor = 1e-4` (NOT the Phase-2 multiplicative `g_opg_lmn=1e-7`); `lam0 ≈ 1e-3` (τ·max_diag=1 after correlation
   scaling; NOT Phase-2's `1.0`). New env `JOLT_OPG_DLAM0/DLFLOOR` (dense-specific).
6. **Harden `conv` (SEV-1):** `conv = (dl<tol) && (maxgy<gtol) && (maxgz<gtol)` — one line at `:3879`, `mgy/mgz` already
   computed at `:3882` (hoist a few lines up). Closes the lam-stall false-conv band. **Downgrade §16.2 "guarantees HOLD" →
   "neutralizes the freeze; HOLD is gate-verified."** Compute Fix-A ρ from the ACTUAL (baseline) step taken.
7. **Fix A + hysteresis latch:** fall back to `|dd|+mu` (y,z) when `lam≥LSAT` (1e3) OR δ non-finite; **latch baseline until
   `lam<LSAT/10`** before re-engaging dense (else re-engage oscillation burns ~4 evalLnL/cycle and RAISES nRej ⇒ fails EFF).
   Self-recovery preserved (baseline accept shrinks lam).
8. **FREEZE_MODEL short-circuits the WHOLE dense solve** (skip assemble/reduce/eig, δ=0) — keeps §10's attribution
   diagnostic zero-cost.
9. **(b) baseline-retry within the backtrack = GATED confirmatory arm only** (`JOLT_OPG_BASERETRY`), NOT shipped-on (doubles
   evalLnL on failing cells, collides with EFF). Use once to prove ON never lands below baseline, with dense/baseline reject
   counters split.

### 18.1 Gate delta (vs §13.1)
- **Phase-3-UNIQUE proof-of-build sentinel** — `JOLT_OPG_STEP` ships in `74b63884` already ⇒ use a NEW dense string
  (`JOLT_OPG_DLAM0` or `[OPGDENSE]`), `strings|grep -c` (measured, not -q).
- **`[OPGSTEP]` → `{dense_tried, dense_accepted, fixA_fired(by trigger), dense_clamp_rate}`.** EFF judged on
  `dense_accepted/dense_tried`. **Honest-failure detector: if `fixA_fired≈every avian iter` and `dense_accepted≈0` ⇒ the
  dense direction is bad too ⇒ PIVOT to the λ_min-DETECTION deliverable** (§16.5 / Nguyen-2018 bimodality).
- **Avian decisive assert:** `dense_accepted>0` AND avian lnL **not below OFF** (the §15 −163 freeze must be gone). Pass/kill.
- **Refactor cross-check:** shared `jolt_opg_reduce` reproduces OPGLMIN `step2k-1 lmin/lmax` to full precision on ≥1 cell.
- RF=0 pre-registered per `λ̂_min/λ̂_max` regime (claim only above the floor). Keep byte-id-OFF (PERFECT, must stay), HOLD
  (no −0.05), CO, `[IRCONV]` λ-not-pinned. tol-ladder ONLY after HOLD.

---

## 19. PHASE 3 GATE = 🔴 SPLIT RESULT (job `174159336`, binary `9b6b4519`, 2026-07-20)

**Verdict line: `🔴 PHASE 3 FAILURE`** (fail-closed: any HOLD/AV breach fails the whole gate). But unlike Phase 2's
*uniform* failure, this is a **5-win / 3-loss split with a single mechanical discriminator**, and the wins are large.

### 19.1 Measured (off disk, `opgp3.o174159336`)

| cell | ON lnL | OFF lnL | Δ nat | iters ON/OFF | dense tried/acc | fixA | clamp | verdict |
|---|---|---|---|---|---|---|---|---|
| dna_r4 | −5706880.4741 | −5706880.4682 | −0.006 | **17/48** | 23/16 | 7 | 0 | ✅ FIX (2.8× fewer it) |
| aa_r4 | −7543893.5340 | −7543893.5340 | 0.000 | **30/44** | 39/29 | 9 | 0 | ✅ FIX |
| dna_r10 | −5706875.1649 | −5706875.3029 | **+0.138** | **30/401** | 36/21 | 38 | 5 | ✅ WIN (13× fewer it) |
| aa_r8 | −7543892.5743 | −7543892.8342 | **+0.260** | **202/401** | 333/202 | 0 | 0 | ✅ WIN (2× fewer it) |
| aa_r10 | −7543892.5742 | −7543892.7877 | **+0.214** | 331/401 | 527/331 | 0 | 0 | ✅ WIN |
| jtt_r8 | −7665094.8011 | −7665096.5008 | +1.700 | 401/401 | 229/18 | 382 | 223 | ⚠️ lnL up, no work-drop |
| dna_r8 | −5706885.5655 | −5706875.2897 | **−10.276** | 143/401 | 118/33 | 121 | **111 (94%)** | 🔴 |
| wag_r8 | −7603878.3778 | −7603871.9890 | **−6.389** | 401/401 | 232/22 | 378 | **226 (97%)** | 🔴 |
| **av_r6** | −11216938.1147 | −11216886.2301 | **−51.885** | 401/401 | **7/1** | **599** | 4 | 🔴 DECISIVE |

Ship invariants ALL still green: **Z byte-id-OFF perfect** (DNA/AA/avian/**partitioned**), **LM** — the shared
`jolt_opg_reduce` refactor reproduces the Phase-1-validated spectrum **exactly** (`lmin=6.238907e-06 lmax=3.585457e+00`),
**CO** carryover present, **HF** honest-failure detector fired correctly (dense IS being accepted; Fix A did not quietly
carry a fake pass).

### 19.2 The discriminator is `dense_clamp`, and it means OVER-amplification

`dense_clamp` counts dense steps that hit a **rate clamp or weight floor** (`gpu_lnl_intree.cu:3922` ← `yzClmp`
`:3908`/`:3921`) — i.e. the solved δ was **too big**, not too small. Every regressing cell is a clamp storm (94/97/97%);
every winning cell has clamp ≈ 0. **This is precisely the blue-team SEV-2b prediction recorded in the code at `:3946`**
("a too-low lam floor lets the spectral floor alone regularize a near-null ⇒ blow-up"). The near-null eigendirections
(avian λ̂ ratio 9e-14 — far below the floor) produce δ_i = a_i/(max(λ̂,1e-4)+lam) that overshoots the bounds.

⇒ **The dense direction is not refuted.** It demonstrably works where the spectrum is benign (5/8 cells, up to 13× fewer
iterations *and* better lnL). What is refuted is the **step-length control** (fixed floor 1e-4 + lam0 1e-3).

### 19.3 Two failure modes, NOT one
- **Clamp-storm** (dna_r8, wag_r8, jtt_r8): dense engages constantly, over-amplifies ~95% of the time, accept-rate
  collapses to 8–28%, lam ratchets up.
- **Early-latch** (av_r6): only **7** dense attempts ever vs **599** fixA fires — the latch fired almost immediately and
  hysteresis (release below `LSAT/10`=1e2) never released (`lam_final=6.58e4`). Avian ran **~99% on the Fix-A fallback**
  and *still* lost 51.9 nats.

### 19.4 🔴 OPEN QUESTION (the load-bearing one) — checked, NOT yet answered
The Fix-A fallback direction is verified-clean by source read: `g_y[c]/(fabs(ddY[c])+mu)` (`:3906`), the canonical form,
no `lam` leak; and reject still does `mu*=4.0` (`:3962`), so its backtracking is functional. **So why does a run that is
99% canonical-baseline lose 51.9 nats?** Two candidate explanations with opposite consequences:
1. the **one accepted dense step** (`dense_acc=1`) landed early and knocked the run onto a permanently worse trajectory,
   after which the latch prevented recovery — benign, fixed by step control; or
2. the **scaffolding itself perturbs** (Gram launch state, conv hardening, the `opgStepOn`-only `clmp` extension at
   `:3921`) — a **defect that would confound the entire Phase-3 verdict**.

**Diagnostic submitted: job `174160067`** (`gems_opg_p3diag.sh`, NO rebuild — reuses the gated binary `9b6b4519` so the
numbers are directly comparable). Arm A is the decisive control: `JOLT_OPG_LSAT=0` ⇒ `denseAct=(lam<0)`=false always
(lam is floored to `dlfloor`>0 at `:3952`), so the dense step never fires while all scaffolding stays live. **A == OFF ⇒
(1); A != OFF ⇒ (2).** Arm B raises the spectral floor 1e-4→1e-2 on the three clamp-storm cells (direct test of
over-amplification); Arm C raises lam0 1e-3→1e-1 on dna_r8 (separates "floor too low" from "trust region too wide").

**Phase-3 source NOT committed** (gate fail-closed). tol-ladder still deferred — a regressing binary shows uniform
regression, so it stays parked until a config HOLDS.

---

## 20. PHASE-3 POST-MORTEM DIAGNOSTIC (job `174160067`, binary `9b6b4519`, 2026-07-20) — §19.4 ANSWERED

No rebuild (reused the gated binary ⇒ directly comparable). Exit 0, 2m28s, **3.7 SU** — these cells are cheap, which
makes this kind of ladder the right instrument.

### 20.1 ✅ ARM A — the scaffolding is INERT. §19.4 resolves to branch (1).
`JOLT_OPG_LSAT=0` ⇒ `denseAct=(lam<0)`=false always (verified: `lam` is floored to `dlfloor`>0 at `:3952`).
Measured `dense_tried=0, fixA_fired=606, latched=1` — the dense step provably never fired.

| | lnL | OFF ref | Δ |
|---|---|---|---|
| avian, dense DISABLED, scaffolding LIVE | −11216886.230 | −11216886.2301 | **+0.0001** |

⇒ the Gram launch, the hoisted reduce, the conv hardening and the `opgStepOn`-only `clmp` extension (`:3921`) **do not
perturb the run**. **The Phase-3 verdict is NOT confounded**, and **Fix A is a genuine no-regression net when it engages
from the start**. Every ON≠OFF difference in §19 is therefore caused by *accepted dense steps* — nothing else.

### 20.2 ✅ ARM B — over-amplification CONFIRMED; the dense direction is SOUND

| cell | floor | lnL | Δ vs OFF | clamp-rate | dense acc/tried |
|---|---|---|---|---|---|
| wag_r8 | 1e-2 | −7603867.486 | **+4.503** | 0% (was 97%) | **400/400** |
| dna_r8 | 1e-2 | −5706875.380 | −0.090 (was −10.276) | 0% (was 94%) | 13/19 |
| av_r6 | 1e-2 | −11216912.609 | −26.379 (was −51.885) | 0% | 1/7 |

**wag_r8 reverses from −6.39 to +4.50 nats with a 100% dense accept-rate.** Raising the floor drove the clamp-rate to
**0% in every cell**. ⇒ §19.2 confirmed: the failure was step *length*, and the dense direction is strong.

### 20.3 ARM C — lam0 is the weaker knob
dna_r8 at `lam0=1e-1` (default floor): −0.163, clamp 0/32. Fixes the clamp storm too, but worse than Arm B's −0.090.

### 20.4 🔴 AVIAN IS A SECOND, DISTINCT MODE — not step length
Still −26.4 at floor 1e-2, same signature as the gate: **7 dense attempts, 1 accepted, latched, `lam_final`≈1e7**.
Combined with Arm A (0 dense steps ⇒ *exact* OFF), the arithmetic is unavoidable: **one accepted dense step costs avian
26–52 nats.** On the most degenerate real cell (λ̂ ratio 9e-14) the dense step is a bad *direction*, not merely too long.
Note both arms are 401-**capped** (neither converges), so this is "how far did you get in 401 iters", i.e. one bad early
step sets the trajectory back — not a converged-optimum comparison.

### 20.5 🔴 CONFOUND recorded (do not over-read Arm B)
`JOLT_OPG_DLFLOOR` is a **COUPLED** knob: `:2850` uses it as the spectral eigen-floor **and** `:3952` uses it as the `lam`
clamp-lo. Arm B moved both at once. Separating them needs a code change (distinct clamp-lo) — deferred to Phase 3b iff
it matters. Read Arm B as "the floor knob as currently wired", not as a pure spectral-floor result.

### 20.6 NEXT — floor ladder, job `174160203` (submitted)
The load-bearing open risk: **a higher floor shrinks EVERY dense step, and all the Phase-3 wins (dna_r10 13×, aa_r8 2×,
both R4 cells) were obtained at floor 1e-4.** Rescuing the regressors could destroy the winners — the "fixed the gate,
broke the product" trap. The ladder therefore measures **both arms at the same floors** (regressors at 1e-1/1e0/1e1,
winners at 1e-2/1e-1), reporting iters as well as lnL. Decision rule pre-registered in the script:
- one floor holds every regressor **and** keeps the winners' work-drop ⇒ that floor is the Phase-3b default;
- winners lose their work-drop as the floor rises ⇒ **floor is the wrong knob** ⇒ fix is a **step-NORM trust-region cap**
  (bound ‖δ‖, leave the spectrum alone) so a single catastrophic step cannot land;
- avian never holds at any floor ⇒ its mode is the one-bad-step/early-latch ⇒ needs the norm cap and/or a
  spectrum-trust precondition, and it remains the gating cell for Phase 3b.

---

## 21. PHASE 3b — DESIGN (step-NORM trust region). Written BEFORE the ladder lands; gated ON it.

### 21.1 Why a norm cap and not just a bigger floor
The floor knob is **indiscriminate**: `δ_i = a_i/(max(λ̂_i,floor)+lam)` — raising `floor` shrinks *every* component,
including the ones that produced the wins (dna_r10 13×, aa_r8 2×). §20.6's ladder exists precisely to measure that
damage. A **step-norm cap is discriminate**: it binds only when a step is actually too large, and leaves every
already-small (good) step bit-unchanged. It also directly targets the *measured* failure — `r=exp(baseY+stepY)` at
`:3907`, so the rate clamp at `:3908` fires exactly when `|stepY|` is large. Bounding `max|dy|` prevents that clamp *by
construction* rather than by hoping a spectral floor happens to shrink it enough.

It is also the standard Levenberg–Marquardt safeguard the current code lacks: today `lam` damps the *solve* but nothing
bounds the resulting **step length**, which is why a single step on avian (λ̂ ratio 9e-14) can cost 26–52 nats (§20.4).

### 21.2 The change (all inside `jolt_opg_dense_solve`, `:974-988` — one function, no call-site change)
Decompose the existing `P` into its two exact model terms (already computed in the same loop at `:981`):
```
G = Σ_i a_i²/den_i           (= gᵀδ, the linear term)
C = Σ_i (a_i²/den_i²)·ev_i   (= δᵀHδ, the curvature term)
P = G − ½C                    ← identical to the shipped formula, refactored not changed
```
Then after `dy`/`dz` are formed (`:985-986`), apply the cap in the **(y,z) infinity norm** (the space that gets
exponentiated, i.e. the space the clamp lives in):
```
m = max( max_i|dy_i| , max_i|dz_i| )
if (m > dmax) { s = dmax/m; scale dy,dz by s; P = s·G − ½s²·C; capped=true; }
```
The rescaled `P` is **exact** for the scaled step (quadratic model evaluated at `sδ`) ⇒ the Nielsen ρ machinery stays
valid and clamp-aware ρ is unaffected. `dmax` default **0.5** (a rate may change by at most `e^0.5`≈1.65× per step);
env `JOLT_OPG_DMAX`, and `dmax=0` (or ≥1e300) disables the cap ⇒ **bit-identical to the current Phase-3 behaviour**,
which is what makes the A/B honest.

### 21.3 Instrumentation
Add `dense_capped` to `[OPGSTEP]` alongside `{dense_tried,dense_acc,fixA_fired,dense_clamp}`. Pre-registered reading:
**`dense_capped` should rise as `dense_clamp` falls to 0** — that is the cap doing its job (converting would-be clamps
into bounded steps). If `dense_capped` is high **and** `dense_clamp` is still high, the cap is not the binding
constraint and the mode is something else — report that, don't tune around it.

### 21.4 Pre-registered gate (Phase 3b), and what would FALSIFY the design
Same matrix as §18.1, plus:
- **Winners must keep their work-drop**: dna_r10 ≤ ~60 iters, aa_r8 ≤ ~250, both R4 cells still < OFF. A cap that
  rescues avian by flattening the wins is a **failure**, not a trade — the whole point of §20.6 is refusing that trade.
- **Regressors must HOLD** (≥ −0.05): dna_r8, wag_r8, jtt_r8.
- 🔴 **avian ≥ −0.05** — still the decisive cell. §20.4 established one bad step costs it 26–52 nats, so avian is the
  sharpest available test of whether bounding step length actually bounds the damage.
- `dmax` disabled ⇒ byte-identical to Phase 3; OPG OFF ⇒ byte-identical to canonical (unchanged ship invariant).
- **FALSIFIER:** if avian still regresses at *any* `dmax` small enough to be a real bound (say 0.05), then bounding the
  step length does NOT bound the damage ⇒ the dense **direction** is wrong on near-null spectra, and the honest move is
  a **spectrum-trust precondition** (refuse the dense step when λ̂_min/λ̂_max is below a threshold — i.e. let the
  *validated* λ̂ diagnostic gate its own solver) or the pivot to the λ_min-DETECTION deliverable.

### 21.5 Standing constraints
Worktree `iqtree3-opg` only (canonical never touched); default-OFF and byte-identical when off; DNA **and** AA; the
partitioned `-p` cell stays in the gate; measured sentinel via `strings|grep -c`; no GPU push; sole-author commits with
no AI attribution; **source commits only after the gate passes.**

---

## 22. FLOOR LADDER (job `174160203`, 13.3 SU, 8m52s) — 🔴 THE FLOOR IS NOT A KNOB, AND §21 IS FALSIFIED

### 22.1 Measured

| cell | floor | Δ vs OFF | iters | dense acc/tried | clamp |
|---|---|---|---|---|---|
| av_r6 | 1e-4 (gate) | −51.885 | 401/401 | 1/7 | 4 |
| av_r6 | 1e-2 | −26.379 | 401/401 | 1/7 | 0% |
| av_r6 | **1e-1** | **−116.180** | 401/401 | 1/6 | 0% |
| av_r6 | **1e0** | **−383.579** | 278/401 | **206/214** | 0% |
| av_r6 | 1e1 | −271.093 | 401/401 | 133/137 | 0% |
| dna_r8 | 1e-1 | −0.233 | 212/401 | 150/160 | 0% |
| **dna_r10** | 1e-4 (gate) | **+0.138** | **30/401 (13×)** | 21/36 | 14% |
| **dna_r10** | **1e-2** | **−10.528** | 401/401 | 400/794 | **99%** |
| dna_r10 | 1e-1 | −0.344 | **29/401 (13.8×)** | 28/33 | 0% |
| aa_r8 | 1e-2 / 1e-1 | +0.259 / +0.216 | **401/401 (no drop)** | 400/400 | 0% |
| aa_r10 | 1e-2 | +0.202 | **401/401 (no drop)** | 400/454 | 0% |
| wag_r8 | 1e-2 / 1e-1 | **+4.503 / +4.290** | **401/401 (no drop)** | 400/400 | 0% |
| dna_r4 | 1e-2 | −0.002 | 18/48 (2.67×) | 17/23 | 0% |
| aa_r4 | 1e-2 | +0.000 | 28/44 (1.57×) | 27/35 | 0% |

### 22.2 🔴 The floor is NON-MONOTONE ⇒ not tunable, not shippable
avian: −51.9 → −26.4 → **−116** → **−384** → −271 as the floor rises. dna_r10: **+0.138** @1e-4 → **−10.5 (99% clamp)**
@1e-2 → −0.34 @1e-1. **A knob whose response is not monotone cannot be tuned, defended, or shipped** — and no single
floor makes every cell hold. Note this also means §20.2's "floor 1e-2 fixes it" was a **local** reading of a
non-monotone curve; the ladder is what exposed that, which is exactly why the ladder came before the build.

### 22.3 🔴🔴 MECHANISM — shrinking the step does NOT bound the damage; it LICENSES it
avian accepted **1/6** dense steps at floor 1e-1 but **206/214** at floor 1e0 — *with the worst loss of the whole
ladder* (−383.6). Smaller steps make a **wrong direction** pass the ρ test, so the optimizer takes *many* small wrong
steps instead of having a few big ones rejected. **On near-null spectra the dense DIRECTION is wrong, not its length.**

⇒ **§21 (step-NORM trust-region cap) is FALSIFIED BEFORE BUILD.** A norm cap shrinks δ exactly as raising the floor did
(it is a milder, direction-preserving version of the same intervention), and the ladder shows shrinkage makes avian
*worse*. Per §21.4's own pre-registered falsifier, this is the kill condition. **§21 is retired unbuilt** — cost: 13 SU
instead of a build+gate cycle. (Caveat kept honest: a norm cap preserves direction whereas the floor rotates δ toward
large-λ̂ components, so the two are not identical — but the ladder gives no evidence shrinkage helps and strong evidence
it licenses accumulation, so building it would be hope, not inference.)

### 22.4 🔴 The §20.6 trap was REAL — the work-drop wins do not survive the regression fix
At the floors where the regressors stop regressing, **aa_r8, aa_r10 and wag_r8 all run 401/401 = NO work-drop.** The
gate's headline 2× (aa_r8) and 13× (dna_r10) drops belong to floor **1e-4** — the same floor that produces −10 to −52 nat
regressions. **The wins and the regressions are the same setting.** Only dna_r4 (2.67×), aa_r4 (1.57×) and dna_r10@1e-1
(13.8×, but −0.344) keep a work-drop, i.e. the *matched-k* cells.

### 22.5 What survives as a real result
- **wag_r8 +4.50 / +4.29 nat, reproduced at two floors, 400/400 dense accepts** — the dense step finds a materially
  better optimum on a real AA cell (no work saved, but a better answer).
- aa_r8 +0.26, aa_r10 +0.20 — same character.
- **Matched-k cells (R4) get a genuine 1.6–2.7× work-drop with lnL held.**
- **Over-parameterized cells are where it breaks** — and that is precisely what [OPGLMIN] already MEASURES.

### 22.6 NEXT (the only hypothesis the data supports): let the VALIDATED diagnostic gate its own solver
§21.4 named the surviving alternative: a **spectrum-trust precondition** — refuse the dense step when λ̂_min/λ̂_max is
below a threshold. Two things make this the right test rather than another tuning attempt:
1. **It is safe by construction**: §20.1 Arm A *proved* that when the dense step never fires, the run reproduces OFF
   **exactly** (+0.0001 nat). So "refuse" has a measured, zero-regression floor — unlike every floor setting above.
2. **It gives the novel λ̂ diagnostic its first concrete job** — gating the solver that its own spectrum says is
   untrustworthy — which is the thesis contribution, not a tuning parameter.

**Decisive precondition: does λ̂_min/λ̂_max actually SEPARATE the winners from the losers?** Known: avian R6 = 9e-14
(loses at every floor), R4 matched = 0.086–0.11 (wins). Unknown and load-bearing: aa_r8/wag_r8 **win** while dna_r8/
dna_r10 **lose** — if those sit on the same side of the ratio, **no threshold exists and the precondition dies too.**
Job **`174160263`** measures the ratio for all 9 cells (pure diagnostic, `JOLT_OPG_LMIN=1`, **no** `JOLT_OPG_STEP` ⇒ the
optimizer runs exactly as canonical, so the ratios describe the UNPERTURBED problem — which is what a precondition must
be computable from). The script computes the separation itself and is written to **kill** the idea: if the winner and
loser ratio ranges overlap, no threshold exists and it says so, with the honest pivot named in the output.

---

## 23. SEPARATOR TEST (job `174160263`, 10.9 SU) — 🔴 NOT SEPARABLE. The spectrum-trust precondition DIES.

### 23.1 Measured λ̂_min/λ̂_max (pure diagnostic, no step change)

| cell | outcome | ratio | λ̂_min | λ̂_max |
|---|---|---|---|---|
| dna_r4 | WIN 2.7× | **8.62e-02** | 1.45e-01 | 1.69 |
| aa_r4 | WIN 1.6× | **1.08e-01** | 1.78e-01 | 1.66 |
| wag_r8 | **WIN +4.50** | **1.98e-06** | 6.98e-06 | 3.52 |
| aa_r8 | WIN +0.26 | **1.74e-06** | 6.24e-06 | 3.59 |
| jtt_r8 | lnL +1.70, no work-drop | **1.65e-06** | 5.87e-06 | 3.55 |
| aa_r10 | WIN +0.20 | 8.18e-10 | 3.69e-09 | 4.51 |
| dna_r8 | **LOSE −10.3** | 3.01e-07 | 1.09e-06 | 3.63 |
| dna_r10 | +0.138@1e-4 / −10.5@1e-2 | 3.90e-11 | 1.79e-10 | 4.57 |
| av_r6 | **LOSE −51.9..−384** | **9.48e-14** | 6.22e-13 | 6.56 |

### 23.2 🔴 The kill is structural, not marginal
**wag_r8 (WIN +4.50) = 1.98e-06 and jtt_r8 (no work-drop) = 1.65e-06 — two AA +R8 cells whose spectra differ by <20%,
with divergent outcomes; aa_r8 (WIN) sits BETWEEN them at 1.74e-06.** Three cells, ratios 1.65/1.74/1.98e-06, outcomes
interleaved. No threshold can cut that.

Worse, the relationship is not even **directionally** consistent: **aa_r10 WINS at 8.18e-10 while dna_r8 LOSES at
3.01e-07** — the winner is 368× *more* degenerate than the loser. Being near-null does not predict that the dense step
hurts.

**Robust to relabelling (checked, because the classification was mine and could flatter the result):** under the
*fairest* labelling — best observed outcome across all floors, which promotes dna_r10 to WIN and jtt_r8 to lnL-WIN — the
only unambiguous losers are dna_r8 (3.01e-07) and av_r6 (9.48e-14), and **dna_r8 still sits in the middle of the winner
range** (three winners above it, two below). Not separable either way.

**Partial rescue only, and it would be curve-fitting:** avian *is* uniquely lowest (9.48e-14, 2.4 orders below the next
cell), so a threshold ~1e-12 would refuse exactly avian — but dna_r8 would still regress −10.3, and tuning a cut to
exclude one dataset is post-hoc fitting on n=1, not a defensible criterion. Rejected.

⇒ **The λ̂ ratio does NOT predict where the dense step helps. The spectrum-trust precondition is dead.** With §21 (norm
cap) falsified by §22 and the floor non-monotone, **every remaining repair route for the Phase-3 solver is now closed.**

### 23.3 🟡 The one pattern that DOES hold — and it is a data-type split, not a spectrum split
**Every AA cell wins on lnL (aa_r4, aa_r8, aa_r10, wag_r8 +4.50, jtt_r8 +1.70). Every DNA high-k cell loses**
(dna_r8 −10.3, dna_r10 −10.5 at the floor that fixes others, avian −51.9..−384); DNA matched-k (dna_r4) wins.
This is a **cleaner separator than the spectrum** — and it echoes the known, independently-recorded DNA-specific
weakness of this stack ([[project_mf_noctf_offload]]: the DNA MF loss is ARCHITECTURAL; AA MF wins ~1.62×).
🔴 **Stated as an OBSERVATION, not a result:** n=4 DNA vs 5 AA cells, one seed, and no mechanism identified. It is a
hypothesis worth one cheap test (more DNA cells, more seeds) — **it is not evidence that an AA-only dense step should
ship**, and it must not be quoted as one.

### 23.4 Honest position
The **solver** does not ship: Phase 2 (diagonal) coupling-blind ⇒ fail; Phase 3 (dense) wins and regressions occur at the
**same** knob position; floor non-monotone ⇒ untunable; norm cap falsified pre-build; precondition non-separable.
What **does** stand, fully validated and unaffected by any of this:
1. ⭐ **The λ_min identifiability diagnostic** (Phase 1 PASSED: independent per-pattern FD vs `patlh` + host-vs-device
   Gram; committed `1bb82e14`). It measures over-parameterization across **12 orders of magnitude** with a clean,
   interpretable signal: matched-k 8.6e-02/1.1e-01 → over-parameterized 1e-06..1e-11 → real avian 9.5e-14. **This
   separator test is itself the strongest evidence yet that the diagnostic is measuring something real and sharp** —
   it just does not predict *solver* behaviour, which was never its claim.
2. The night3/night4 result: tight tol is CORRECT, loose tol reproducibly WRONG (spurious +I), and GPU-tight beats a
   full CPU node on **both** speed (2.9×) and correctness.
3. The negatives themselves: the diagonal Fisher is coupling-blind; the dense step's failure is **direction, not
   length**; and **shrinking a step licenses a wrong direction rather than bounding it** (avian 206/214 accepts at its
   worst loss) — a transferable lesson about trust-region damping on near-singular empirical-Fisher blocks.

**Recommendation: stop solver work; ship the diagnostic.** This is a scope decision for the author, not mine to take —
no further build proceeds without a discrete greenlight. Phase-3 source remains UNCOMMITTED.

---

## 24. CONCLUSION — solver closed, diagnostic ships (greenlit 2026-07-20), externally corroborated

### 24.1 The verdict
The OPG-conditioned +R **solver does not ship**. Every route was tried and closed, on real data:
- **Phase 2 (diagonal OPG)** — coupling-blind → avian −163 nats (§15).
- **Phase 3 (dense (2k−1) spectral)** — 5 wins / 3 losses, but the wins and the regressions live at the **same knob
  position**; the failure is over-amplification of near-null eigendirections (§19).
- **Repair routes** — floor is **non-monotone** (untunable, §22); the step-**norm** cap was **falsified before build**
  (shrinking a step *licenses* a wrong direction rather than bounding it, §22.3); the λ̂-ratio **precondition is
  non-separable** (winners and losers interleave: aa_r8/wag_r8 WIN at ~1.7e-6 while dna_r8 LOSES at 3.0e-7, §23).

**What ships instead: the λ_min identifiability diagnostic** (Phase 1 PASSED, committed `1bb82e14`; independent
per-pattern FD vs `patlh` + host-vs-device Gram). It measures over-parameterisation cleanly across **12 orders**
(matched-k 1e-1 → over-param 1e-6..1e-11 → real avian 9.5e-14). It does not predict *solver* behaviour — that was never
its claim — but the separator test (§23) is itself the strongest evidence it measures something real and sharp.

### 24.2 External corroboration (Hashara BFGS, job `174165296`) — the decisive independent check
The shared cross-check (`README-avian-convergence.md`) asked whether a true quasi-Newton **BFGS** (the off-diagonal
curvature our diagonal-LM discards, and the OPG dense solve *tried* to capture) stabilises the avian +R winner where
ours flips. We ran her OpenACC binary ourselves. **It does NOT**: seed1 `GTR+F+I+R5`, seed2 `GTR+F+I+R6` — flips k AND
carries the same **spurious +I**, worse BIC than ours at the same epsilon. ⇒ **the avian +R instability is the
DATA/EPSILON, not any one optimiser.** No method — diagonal-LM, OPG dense, or a real BFGS — stabilises it at 0.1; only
tight tol (1e-7 → `GTR+F+R6` ×3, no +I) does. This is direct external proof the solver route is a dead end and the
DETECTION diagnostic is the right deliverable. (Speed aside: her 12t avian MF 2041–3594s vs ours@0.1 ~130–150s ≈ 15–27×.)

### 24.3 night2 (job `174157172`) — the tight-necessity boundary is DATA-TYPE-SPECIFIC
Real AA high-k (euk `CAT_100S93F`): tight **and** loose agree every seed (`LG+(F)+I+R9`); k is stable (R9 ×3), +I is
genuine (kept by tight), the only wobble is a tol-independent ±F near-tie on seed 1. **Tight buys nothing here and costs
~10×.** ⇒ "tight is necessary for correctness" is **DNA/avian-specific**; **AA is tol-robust**. +R = **99.1%** of the AA
MF wall (140 at the 401 cap) — so the cap tax is real on AA, but loose already gives the right AA winner, so the OPG's
"make tight fast" goal never applied to AA. Net: reinforces solver-dead — no optimiser helps where tight is *needed*
(avian), and where tight isn't needed (euk AA) loose is already fast+correct. Evidence FOR data-type-aware tol, AGAINST
any blanket global tol.

### 24.4 What is now moving to the shared binary (promotion, greenlit)
The λ_min diagnostic + the gate-passed **MFVAL** self-check are being promoted together (combined worktree
`iqtree3-promote` @ `163c2dc9`, zero file overlap) through the **P2 sweep gate** — red-team + blue-team audited (the two
balanced: red tightened Z against false-PASS, blue caught the tightening's avian false-BLOCK edge). Gate job
`174174353`; PASS → author does P3 (canonical merge → rebuild → `GPU-BINARIES.md` → repoint `iqtree3-gpu-latest` → push
`gpu-kernel-dev`). Assistant CERTIFIES only; no push. The +R **tolerance** win remains un-shippable (global 1e-2 dead;
screen+repolish retired; make-tight-fast/OPG failed) — the sole optimisation still outside the binary with no safe form.

**Dead solver preserved** as `opg-phase23-DEAD-SOLVER.patch` (reproducible negative). Phase-3 source UNCOMMITTED.
