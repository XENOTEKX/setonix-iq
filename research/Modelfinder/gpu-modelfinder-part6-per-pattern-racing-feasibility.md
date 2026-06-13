# PART VI — Feasibility: streaming per-pattern + racing/pruning ModelFinder for IQ-TREE GPU

**Author:** as1708 / research synthesis (Claude, workflow `wl481i0v8`: 6 gather agents — internal docs, IQ-TREE
source, 3 internet literature sweeps, quantitative grounding → synthesis → 4 adversarial red-teams → verdict),
2026-06-10. Ultracode run.
**The idea assessed (user's):** replace per-MODEL parallelism with streaming per-PATTERN processing — run all
~224 candidates concurrently on small pattern chunks, keep a running gradient + running lnL per model, and PRUNE
(successive-halving / racing) the under-performers early, so the winner ends up evaluated on all the data.
Motivation: keep each model's working set small to **escape the memory-bandwidth bound** and **fill the GPU**.

---

## VERDICT: UNLIKELY against the stated bar (beat FCA on 16 Sapphire-Rapids nodes at 1M-AA ModelFinder)

Two of four adversarial red-teams were **FATAL** on the GPU-beats-FCA claim, and they are dispositive:

### Fatal 1 — the bandwidth-escape premise is FALSIFIED by our own P3.0
The whole GPU motivation ("shrink the working set to escape the bandwidth bound") rests on the kernel being
bandwidth-bound. **It is not.** P3.0 (job 170398260) measured DRAM **flat at ~33% (308/900 GB/s) from 100K→300K
patterns — no climb toward saturation**; P3.0b measured `stall_long_scoreboard` 50% (memory **latency**),
`math_pipe` 0.08% (≈no compute), schedulers issuing on <9% of cycles. The kernel is **latency/occupancy-bound.**
Consequences:
- "Escape the bandwidth bound" relieves pressure that was **never the limit.**
- The correct lever for a latency bound is **more resident warps (occupancy)** — but **chunking one model
  *lowers* its block count and thus its occupancy: it moves the wrong way.**
- Chunking is **byte-neutral-to-worse**: because `lnL = Σ_p w_p·logL_p` with per-pattern-independent partials,
  every pattern's array is touched exactly once per sweep regardless of chunk size; chunking only **adds**
  per-chunk eigensystem/constant reload overhead.

### Fatal 2 — the breadth-vs-depth N/S ceiling (mechanism-independent)
A mutex-serialized single GPU at per-model speedup S≈4.8 processes candidates one at a time; a 103-core node runs
N≈103 concurrently. **Aggregate = N/S ≈ 21× slower at ANY coverage** (measured TESTONLY: GPU 3493 s vs CPU
~259 s = **13.5×**). No subsample size, pruning aggressiveness, or scale changes that **each surviving model's
full-data refine is block-saturated and runs at one-node speed on a serialized device.** The 1M "~571 s < 1122 s"
figure is **not** a counterexample: CTF is CPU-portable — a 16-node cluster runs the same coarse-rank then refines
top-k *concurrently*, so it is an **algorithm** win over the old algorithm, not a hardware win.

### Serious — the "running gradient" half is unsound; stream only the RANKING
Stepping the optimizer on partial-pattern sums divides a noisy chunk-`df` by a noisy chunk-`ddf` (chunk curvature
can flip sign), **destroys JOLT's full-data accept/reject safeguard**, provably **loses superlinear convergence**
(needs noise→0), and **lengthens the latency-bound dependent critical path** the idea meant to fill — breaking
JOLT's validated same-MLE contract (cold==warm rel 2.5e-16). SGD/mini-batch only wins for **large-redundant-data,
many-parameter, non-convex** problems; IQ-TREE's MLE is the **opposite** (patterns pre-compressed to unique sites,
p≈200, smooth, converges full-batch in 14–27 sweeps). **Confine chunking to exact lnL accumulation at FIXED
parameters for the ranking decision only**; hand survivors to the unmodified full-batch JOLT/CPU optimizer.

### The racing/pruning half IS sound — but it's CTF, and CPU-portable
- `lnL = Σ_p w_p·logL_p` is a sum of **bounded** per-pattern terms ⇒ subsample lnL is a concentration-bounded
  estimator (Hoeffding-races regime); race the **paired per-site lnL difference** (the RELL trick already used by
  phylogenetics' KH/AU tests) for a domain-native, variance-aware criterion.
- **Published precedent (CPU) — VERIFIED:** *ModelTamer* — Sharma & Kumar, "Taming the Selection of Optimal
  Substitution Models in Phylogenomics by Site Subsampling and Upsampling," *Mol Biol Evol* 39(11):msac236, Oct
  2022 (PMID 36306418). Selects the correct substitution model **hundreds–thousands× faster** from "a small
  representative fraction of unique site patterns," with **upsampling** (≡ the scale-consistent-BIC correction
  below — apply the full-data site count) as the key fix. This is **literally the "rank-on-subsample, winner sees
  all the data" idea, on the CPU, already published.** *(The VERDICT does NOT depend on this citation: the two
  FATAL findings are our own measurements — P3.0 bandwidth falsification + the N/S ceiling — so the "no" stands
  even if ModelTamer evaporated. ModelTamer only reinforces that the surviving half is CPU prior art, not novel.)*
- **Required correction:** BIC's `k·ln(n)` penalty is n-dependent, so ranking on small n **under-penalises
  parameter-rich models.** Must use **scale-consistent BIC** `BIC' = −2·(N/m)·lnL'_sub + p·ln(N)` (full-data N in
  the penalty), which CTF already specifies.
- **Offline kill-switch (free, run 2026-06-10 on the existing `ctf_p0/rank_{1000,2000,5000}.iqtree`):** with
  scale-consistent BIC, LG+G4 ranks **#1 at every K**; LG+I+G4 stays **#2 within ΔBIC′ +25…+65** (thin-margin
  runner-up survives); LG+F+G4 #3. **PASS** — but **passes trivially** on P0's atypically clean data (17,618-nat
  cliff to #4, and **no +R/+I+R** in TESTONLY) ⇒ **necessary-not-sufficient.**

### The genuinely novel sliver — and it's still CPU-portable + unvalidated
The only net-new component is the **adaptive multi-round racing controller** (drop laggards having seen *fewer*
chunks than survivors, with anytime-valid confidence sequences). It needs **no new CUDA kernels** (existing
per-pattern-additive kernels wrap in an outer chunk loop), it is **algorithmic and CPU-portable** (nothing
GPU-causal), and it is **unvalidated** — exactly the deferred "speculative pre-emption" gap
(`novel-dispatch-architectures.md`: "needs theoretical work on BFGS/BIC lower bounds"). P0's good recall validates
only *fixed-subsample full re-ranking* (= two-stage CTF), **not** adaptive early-dropping. The realistic casualty
is a thin-margin, parameter-rich slow-starter (LG+I+G4 at ΔBIC≈14; or a +R/+I+R model whose edge rides sparse
high-variance rate classes).

---

## What survives, and is worth having

A **CPU-portable adaptive/finer-grained CTF + the already-built JOLT** (4.8×/model, measured) **⏳ projects** a
**~1.5–7× speedup over the STOCK IQ-TREE tool at 100K** (~57–151 s vs the 221 s AVX-512 floor / 399 s `-m MFP`)
for users on **a single device without a cluster.** ⏳ = **PROJECTED, NOT measured** — it rides on warm-start
refine-iteration counts never measured (the part5 P1 gate); only JOLT's 4.8×/model is measured. That is
**accessibility value, not GPU supremacy** — and it does **not** beat FCA-16-node at any scale.

## Honest bottom line for the user's bar
**No.** The idea cannot beat a 16-node CPU cluster at 1M-AA ModelFinder: the breadth-vs-depth ceiling is
mechanism-independent (≈21× at any coverage), the bandwidth-escape premise is falsified (the kernel is
latency-bound; chunking moves the wrong way), and the genuinely-new part (adaptive racing) is itself
CPU-portable. ModelFinder is a breadth/dispatch problem the cluster owns by design; nothing in streaming or
pruning changes that the surviving model's full-data refine is one-node-bound on a serialized GPU.

## If pursued anyway (only as a single-device tool-speedup, not a cluster-beater)
1. **P-kill (free, DONE):** offline halving sim on existing `rank_*.iqtree` + scale-consistent BIC → PASS (trivial).
2. **P-kill2 (small new run, decisive):** repeat adaptive-drop recall on ONE *less-separated* alignment + a
   candidate set **including +R/+I+R** at a few subsample sizes. Gate: true winner survives every rung; no complex
   true-best pruned early. Fail ⇒ adaptive dropping unsafe ⇒ fall back to two-stage CTF.
3. **P1:** exact chunked lnL accumulation at FIXED params (outer chunk loop on existing kernels) + scale-consistent
   BIC. Gate: chunked lnL == one-shot to machine precision; reload overhead doesn't make it byte-worse.
4. **P2:** adaptive racing controller → survivors to unmodified JOLT/CPU refine. Gate: reproduces LG+G4 AND is
   measurably faster than plain two-stage CTF (else drop the controller — ModelTamer already recovers the right
   model from a small subsample with no racing at all, so the adaptive sliver must *earn* its complexity).
5. **P3:** honest writeup — single-device adaptive-CTF + JOLT ~1.5–7× over the stock tool; explicitly NOT
   GPU-beats-FCA at any scale.
