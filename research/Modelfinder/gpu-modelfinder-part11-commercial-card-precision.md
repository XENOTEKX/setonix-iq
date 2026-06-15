# PART XI — Commercial-card support: the FP64 precision question (can ModelFinder run on an RTX 5090?)

**Author:** as1708 (multi-agent research synthesis by Claude Opus 4.8, 2026-06-15)
**Status:** RESEARCH VERDICT + DESIGN. Produced by a 2-agent workflow (GPU-hardware/unlock-feasibility + numerical
extended-precision) + a first-principles BIC-gap analysis on the project's own AA data.
**Scope:** the IQ-TREE panel's question — *could JOLT+CTF run 1M-site AA ModelFinder on a consumer card (RTX 5090,
AMD RDNA) by going FP32, given consumer cards lack fast FP64 and ECC — without destroying correctness or IQ-TREE's
reputation?* Motivation: democratise GPU ModelFinder to researchers without $15k datacenter GPUs.

> **One-paragraph verdict.** "Unlocking" FP64 on a modern consumer GPU is **impossible — it is silicon-level market
> segmentation, not a software lock** (the FP64 ALUs are not on the die). But the goal is achievable by **routing
> around the limit, and the route-around is faster than the thing we'd unlock**: on a 1/64-rate card, software
> **double-float ("df64") / compensated reductions on the abundant FP32 units run ~3–6× faster than the card's own
> native FP64**, at near-FP64 accuracy. The correct design is **precision-tiering**, not a wholesale FP32 switch:
> plain FP32 for the partial-likelihood recursion, explicit **scaling** for underflow (the real FP32 danger — not
> mantissa), a **compensated/df64/long-accumulator reduction** for the BIC-critical cross-site lnL and gradient sums,
> and native FP64 for the ~200 optimiser scalars. Net overhead ≈ a few %, FP64-equivalent BIC. Shipped **opt-in**,
> with the final winner **re-verified in native FP64** and the delta reported (never silent) and a fixed-order/long
> accumulator for cross-card bit-reproducibility, this **democratises ModelFinder to consumer cards without risking
> wrong science or IQ-TREE's reputation.** `+R` stays FP64 (dynamic-range overflow); the datacenter path (A100/H200,
> 1/2-rate FP64) is **unchanged** — native FP64 wins there.

---

## XI.1 The question and why it matters

A1/H100/H200 cost $10–40k; an RTX 5090 costs ~$2k and a used 3090/4090 far less. If ModelFinder ran on a consumer
card, any lab — regardless of budget — could use the GPU pipeline. The blocker the panel named: consumer cards have
**crippled FP64** and **no ECC**, and our entire correctness story (bit-parity lnL, FD-validated gradients) is FP64.
The question has three parts: (i) can the FP64 throttle be removed? (ii) if not, can we get FP64-equivalent results
in FP32? (iii) does doing so endanger IQ-TREE's reputation for correct, reproducible science?

## XI.2 Can FP64 be "unlocked"? — NO. It is silicon, not software.

**Verdict: impossible on any modern consumer card.** The FP64 deficit is **die-level segmentation** — the consumer
dies physically ship with ~1 FP64 ALU per 64 FP32 lanes; the datacenter parts are *different silicon* with full-rate
FP64. NVIDIA's own Ada whitepaper states AD102 has **288 FP64 cores (2/SM) present only "to ensure FP64 code operates
correctly"** — a correctness provision, not throughput. There is no high-rate mode to un-gate.

| GPU (die) | Arch | Segment | FP64:FP32 | FP64 (≈) |
|---|---|---|---|---|
| A100 (GA100) | Ampere | DC | **1/2** + FP64 tensor | ~9.7 / 19.5 TF |
| H100/H200 (GH100) | Hopper | DC | **1/2** + FP64 tensor | ~34 TF |
| RTX 3090 (GA102) | Ampere | consumer | **1/64** | ~0.56 TF |
| RTX 4090 (AD102) | Ada | consumer | **1/64** | ~1.3 TF |
| RTX 5090 (GB202) | Blackwell | consumer | **1/64** | ~1.6 TF |
| **B300 / Blackwell Ultra** | Blackwell | **DC** | **1/64 (!)** | ~1.2 TF |
| RX 7900 XTX (Navi 31) | RDNA3 | consumer | **1/32** | ~1.9 TF |
| RX 9070 XT (Navi 48) | RDNA4 | consumer | **1/64** | ~0.76 TF |
| MI250X / MI300X (CDNA2/3) | CDNA | DC | full-rate | ~48 / 82 TF vector |

**History (why the precedent doesn't apply):** the Kepler GTX Titan/Titan Black (GK110, 2013) had a driver toggle
("CUDA – Double Precision") enabling ~1/3-rate FP64 — *only because the die physically had the units*, merely
clock-gated. AMD Radeon VII (Vega 20) had 1/4. That silicon era ended; Ada/Blackwell/RDNA4 removed the units, so the
toggle has nothing to act on. **Cross-flashing** a consumer card to a Quadro/datacenter identity gives **zero** FP64
benefit (RTX 4090 and RTX 6000 Ada are the *same die*); flashing to an A100 identity is physically impossible
(different die). **Two buying-decision facts:** AMD consumer FP64 is *regressing* (1/16→1/32→1/64), and even
datacenter FP64 is eroding (B300 = 1/64) as AI revenue makes FP64 segmentation moot.

**Legal/reproducibility:** BIOS/driver hacks void EULA/warranty, risk bricking, and — decisively for a tool shipped
under IQ-TREE's name — produce a **non-reproducible substrate** (a reviewer cannot reproduce "RTX 4090 + hacked
vBIOS"), with elevated risk of *silent* numerical error (a wrong tree, not a crash). Unlocking is a dead end.

## XI.3 The engineering route-around: emulate precision on FP32 — and it beats native FP64

Error-free-transformation (EFT) arithmetic represents one value as a sum of FP32 words:

| Technique | Accuracy | Cost / op | Note |
|---|---|---|---|
| **double-float "df64"** (hi+lo FP32) | ~48-bit mantissa (FP64=53) | **~10–20 FP32 ops** | FMA two-product is 2 ops on a GPU (every FP32 unit has FMA) — the enabler |
| double-double (Bailey QD, FP64 words) | ~106-bit | DD add 6 / mul 8 FP64 ops | for FP64 hardware; not the consumer case |
| **compensated sum** (Kahan/Neumaier) | recovers the √n loss → ~FP32 ε in the *result* | ~4 FP32 ops/term | the cheap win for a *reduction* specifically |
| pairwise/cascaded reduction | O(ε log n) | ~1 op/term | GPU-natural (a reduction tree already does this) |
| ExBLAS long accumulator | **bit-exact, reproducible** | high | the gold standard for *reproducibility* |

**Net speed (the decisive arithmetic).** On a card where FP64 = FP32/64, one native FP64 op costs ≈ 64 FP32 ops; one
df64 op costs ~10–20. **⇒ df64 is ~3–6× FASTER than native FP64 on a 1/64 consumer card**, at ~48-bit accuracy. The
crossover is "FP64:FP32 ratio vs ~10–20": **consumer cards (1/64, even 1/32) → emulate; datacenter (1/2) → native FP64
crushes df64.** So this is *specifically* a consumer-card port; A100/H200 keep native FP64 unchanged.

**Ozaki / tensor-core FP64 emulation is a red herring here.** It achieves FP64-accurate GEMM on INT8/BF16 tensor
cores and is brilliant for large dense matmul — but phylo's hot loop is tiny per-edge matrix–vector + element-wise
products, and the accuracy-critical step is a *reduction*, which Ozaki does not address.

## XI.4 Precision-tiering for phylogenetic likelihood (the heart of the design)

Map each stage to its true precision need — do NOT blanket-emulate:

| Stage | Real risk | Precision (consumer card) |
|---|---|---|
| Partial-likelihood recursion (transition-matrix × child partials, up the tree; **99% of FLOPs**) | error grows ~depth·ε; benign per node | **plain FP32** (full speed) |
| **Underflow scaling** | **FP32's 8-bit exponent (~1e±38)**, NOT mantissa — partials underflow far sooner; df64/DD do NOT help (same small exponent) | **explicit integer/log scaling** (SAFE_LH / scale_log). **#1 correctness item; re-validate on FP32.** |
| **Cross-site lnL sum** (~1e6 terms → ~−7.8e7) | naïve FP32 loses ~4 nats → **flips BIC** (model selection) | **compensated (Neumaier) or df64 / long-accumulator** |
| **Gradient reduction** (Σ over sites; JOLT all-branch + rate) | same; per-site `qp/L_p` ratios self-cancel (G.4.0b) → FP32-safe per-site, the *sum* is sensitive | **same compensated/df64 reduction** |
| Optimiser scalars (~200 branches + α; LM step) | negligible cost | **native FP64** (cheap even at 1/64; protects "same MLE") |

The extended-precision tax lands only on the O(n) reductions + O(200) optimiser vector — a negligible fraction of the
O(n·depth·cats·states) total. **Net slowdown vs all-FP32 ≈ a few %, while the BIC is FP64-equivalent.** This connects
to G.4.0b: the +R path reaches 1/L_ptn ≈ 1e92 dynamic range — **`+R` overflows FP32 and must stay FP64 (or log-space);
the FP32 danger is dynamic range, addressed by scaling, not by extended-precision mantissa.**

## XI.5 Where FP32 is genuinely dangerous — the BIC-gap analysis (real project numbers)

ModelFinder selects by **BIC differences**, so the precision requirement is set by the smallest gap that must be
resolved. Measured on our AA data (winner vs runner-up — the tightest, most dangerous decision):

| Scale | LG+G4 BIC | LG+I+G4 BIC | **ΔBIC** | lnL difference |
|---|---|---|---|---|
| AA-100K | 15086278.74 | 15086290.25 | **11.5** (≈ ln 100000) | 0.0006 nats |
| AA-1M | 157213286.75 | 157213300.57 | **13.8** (≈ ln 1e6) | 0.0047 nats |

**Reassuring structure:** the top-2 likelihoods are essentially *tied* (the +I param is dead); BIC separates them by
the **penalty `k·ln N`, which is computed from integers and is exact regardless of FP32/FP64.** And because the two
models are nearly identical, their FP32 rounding errors are **highly correlated and largely cancel in the difference**
→ within-family near-ties are penalty-protected + error-cancelling. Cross-family decisions are protected by the
**17,618-nat cliff** ≫ any plausible FP32 error. The genuine danger zone is only *moderate-gap* pairs (a few nats)
with *uncorrelated* errors — narrow, and exactly what the compensated/df64 reduction (≪ 1-nat error) covers.

**Literature concurs with caution + the tree-search/selection split:**
- BEAGLE has always supported single precision; it is "sufficient for **tree searches**" but with more scaling
  overhead (Ayres et al. 2019, *Syst. Biol.* — BEAGLE 3).
- The RAxML group (Berger & Stamatakis 2010) found SP fine for most topologies but lnL **diverges from DP on
  large/divergent alignments** (accumulation grows with taxa and tree depth — our height-42 ladder is the watch-item).
- The **BEAGLE 4.0 tensor-core group deliberately chose FP64** "due to the complexity of the phylogenetic likelihood
  surface, characterized by islands of high likelihood" (Gangavarapu et al. 2026) — domain experts avoiding low
  precision for fine inference. The recurring distinction: SP is fine for *topology*, risky for *fine
  estimation/selection* — which is exactly why CTF's exact-BIC refine + compensated reductions matter.

## XI.6 Reproducibility and reputation — the non-negotiables

IQ-TREE's reputation rests on correct, reproducible likelihoods + selections. Risks of a careless FP32 release:
silent wrong selection on moderate-gap pairs; non-reproducibility (FP32 results vary with reduction order / GPU /
driver); rare un-caught bit-flips (no ECC). All are addressable **without** abandoning the consumer-card goal:
- **Opt-in only** (`--mixed`/`--fp32`, OFF by default) with a documented accuracy envelope.
- **Always verify:** the winning model's lnL **re-evaluated in native FP64** (one model — cheap even at 1/64) and the
  FP32↔FP64 delta reported; a **BIC-gap guard** escalates a near-tied top-2 to FP64.
- **Bit-reproducibility:** a fixed-order or ExBLAS-style long-accumulator reduction → order-independent, cross-card
  bit-stable sums (so a 5090 and an H200 agree).
- **Never silent.** The user is informed; the answer is verified; IQ-TREE never publishes a silently-wrong selection.

## XI.7 Recommended design

1. **Don't use native FP64 on a consumer card** (1/64 = catastrophic). Don't BIOS-hack (impossible + non-reproducible).
2. **Tier precision** (§XI.4): FP32 recursion + scaling for underflow + compensated/df64 reduction for the two sums +
   native FP64 for the optimiser scalars. ≈ few-% overhead, FP64-equivalent BIC.
3. **Wrap it in CTF:** coarse rank on the 5000-site subsample in **plain FP32** (√5000-tiny error, top-k safety net);
   refine top-k with FP32 recursion + **df64/long-accumulator reductions**; final winner **re-verified in native FP64**.
4. **`+R` stays FP64/log-space** (dynamic-range overflow). The **datacenter path is unchanged** (native FP64 wins at 1/2).
5. **Opt-in + verify + reproducible accumulator** (§XI.6).

## XI.8 The decisive experiment (to turn this from argument into a measured envelope)

Mirror the G.x methodology: a standalone harness computing full-data lnL + gradient three ways — (a) native FP64
oracle, (b) FP32 + Neumaier reduction, (c) FP32 + df64 reduction — at AA 10K/100K/1M, gated **not** on a mantissa-bit
count but on **|ΔlnL| ≪ 1 nat AND identity of the top-k BIC ranking** (the same recall gate CTF already uses). That
directly measures "does it preserve *model-selection* accuracy" and pins the safe sequence-length ceiling. The FP64
oracle already exists (the cross-scale sweeps). Escalation path if a run drifts: promote *only* the recursion
accumulation (not the products) to df64 — the cheapest fix.

## XI.9 Verdict

**Pursue commercial-card support.** It is genuinely valuable for access-equity and it is achievable **without
unlocking anything and without risking wrong science**: smart precision-tiering on FP32 (df64/compensated reductions +
scaling) is *both* faster than a consumer card's native FP64 *and* reproducible enough to protect IQ-TREE's name. The
panel's "impossible" is true only of the *unlock*; the *route-around* is the answer. The honest residual risks —
df64's 5-fewer mantissa bits, FP32 exponent-range underflow, accumulation through deep ladders, and reproducibility of
parallel reductions — are all bounded and addressable, and the decisive experiment (§XI.8) converts the argument into a
measured accuracy envelope before any public release.
