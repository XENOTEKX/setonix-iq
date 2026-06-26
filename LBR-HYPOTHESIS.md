# LBR — Localized Batch Reopt: a grounded, falsifiable hypothesis

**Date:** 2026-06-26. Tree: `iqtree3-l2search` (single-GPU optimization track; GPU×MPI hybrid deferred).
Modeled on the CTF subsample-recall hypothesis: a concrete mechanism + a quantified prediction + a hard
falsification gate + an honest prior that it might fail.

---
## 0. The target (code-grounded)
The GPU all-branch reopt `gpu_jolt_optimize` (`tree/gpu/gpu_lnl_intree.cu:1643+`) is **fully
non-incremental**: each of its ~12 LM iterations runs a **full postorder partial fill over all `nInternal`
nodes** (`:1826-1830`) **plus a full preorder gradient sweep over all edges** (`:1949-1968`). The only thing it
ever skips is `rebuildEchild` when branch lengths are unchanged (`:1932-1935`) — never the sweeps. The CPU's
lazy `partial_lh_computed` incrementality (`phylonode.h:156`, `clearReversePartialLh` `phylonode.cpp:22-33`)
is **not used** on the GPU path. So per accepted NNI round the GPU spends ≈ `12 × (full tree sweep)`.

## 1. The naive version is FALSE (red-teamed up front)
A tempting LBR = "only recompute the partials an NNI marked stale." **Refuted by the invalidation
semantics.** After an NNI at central edge (node1,node2), `clearReversePartialLh` recurses *outward from both
ends* (`phylotree.cpp:4148-4149`), marking the **reverse/upper partials of the entire tree** stale except the
`O(depth)` ancestors of the edge (whose subtree still contains the edge, so their "outside" is unchanged).
⇒ **one NNI dirties `O(n)` upper partials.** A batch dirties even more. So the post-NNI sweep is *unavoidably
full*; you cannot save by skipping stale-flagged partials. (The grounding agent's "`O(log n)` path-to-root,
2–3×" was exactly backwards — it described the small *valid* set, not the large *stale* set.)

## 2. The REAL hypothesis (where the saving actually is)
The opportunity is not in the stale-set; it is in **which branch OPTIMA actually move.**

> **LBR HYPOTHESIS.** After an accepted node-disjoint NNI batch, the optimal lengths of branches *far from
> every applied move* change by less than the optimizer's own tolerance δ. Therefore the all-branch LM reopt
> can **hold far branches fixed and re-optimize only the affected neighborhood** (branches within graph
> distance d of an applied move, plus the `O(depth)` connecting partials), across the LM iterations — and a
> **single exact full reopt at the very end of the search** recovers the same final tree (lnL within ε,
> RF = 0) as full all-branch reopt every round.

Mechanism it leans on: a branch's ML length is dominated by the *local* likelihood surface (its own
lower×upper partials); a distant rearrangement perturbs that surface at second order. This is the
branch-length analogue of CTF's "the ranking is preserved on a subsample" — here, *the converged branch
lengths are preserved under a localized reopt.*

## 3. Quantified prediction + falsification gate
Let `af` = fraction of the tree's branches whose optimum moves by more than δ during a full all-branch reopt
of a round (the *materially-affected* set). Full reopt = `G` sweeps (`G`≈12). LBR ≈ 1 mostly-localized
initial pass + `(G−1)` sweeps restricted to `af`-fraction of edges, plus an amortized final full reopt:

```
   speedup_per_round  ≈  G / (1 + (G−1)·af)        (≈ 1/af for small af, G large)
   af = 0.10 -> ~5.7x | 0.20 -> ~3.7x | 0.33 -> ~2.7x | 0.50 -> ~1.9x | 1.0 -> 1.0x
```

**GATE (both required to proceed past the spike):**
 - **(G1) Locality:** measured `af ≤ 0.30` (median over rounds) on a real AA-100K search ⇒ ≥ ~2.7× headroom.
   If `af` is large (distant NNIs perturb most optima > δ), LBR is FALSIFIED — the work is genuinely global.
 - **(G2) Exactness recoverable:** a search that holds far branches fixed each round + ONE final full reopt
   reaches a final lnL within ε (≤ 1e-6 rel) and RF = 0 vs the full-reopt-every-round baseline. If the held-
   fixed drift accumulates over the hundreds of rounds and the final full reopt cannot undo it (or needs many
   full sweeps to undo it, eating the saving), LBR is FALSIFIED on quality, not speed.

## 4. Honest prior (realistic, not boosterish)
- **Plausible** that `af` is small: branch optima are local; this is the same physics that makes nni5 (5-branch
  local reopt) and the exact screener work. If those are local, the *converged* lengths should be too.
- **Two real risks that could falsify it:**
  1. **Accumulated drift.** Holding far branches fixed for ~hundreds of rounds may let small per-round errors
     compound; the single final full reopt then has to undo a large accumulated displacement — possibly many
     full sweeps, eroding the win. (Mitigation knob: a *periodic* full reopt every K rounds; cost model becomes
     `G/(1+(G−1)·af) ` amortized with a `1/K` full-sweep tax.)
  2. **The forced initial pass.** Even iteration 1 of a round needs the upper partials on the path from the
     root through the applied moves refreshed (`O(depth)` not `O(1)`); on a caterpillar topology `depth≈n`, so
     the "localized" pass is not local. Balanced/realistic ML trees (depth ≈ log n) are favorable; pathological
     topologies are not. `af` and depth are both topology-dependent ⇒ must be MEASURED on real trees.
- **Ceiling honesty:** even at `af=0.1` the per-round speedup is ~5.7×, but it applies only to the *reopt*
  surface (the dominant cost, ~83% of fused wall) and is bounded by the final-reopt tax and Amdahl on the
  screener. A realistic end-to-end expectation is **2–4×** on the reopt, *if both gates pass*. This is a
  single-GPU, per-iteration compute-reduction lever (it makes one GPU search faster); it is NOT the search
  axis and does not by itself beat MPI — but per the user's plan, that is the right thing to chase first.

## 5. The cheap measurement (decides the gate before any GPU build)
Like the L2.0 spike and the site-repeats probe: measure first, build only on a pass.
 - **Measure `af` (G1):** instrument the existing all-branch reopt (CPU `optimizeAllBranches` or the JOLT path
   with a host dump) to record, per accepted round, every branch's `|Δlen|` and its graph distance to the
   nearest applied NNI. Report the distribution of `af` (fraction with `|Δlen|>δ`) and the affected-vs-distance
   curve, on the real AA-100K tree. Cheap: one instrumented `--ts-fused` run, no new kernel.
 - **Test exactness recoverability (G2):** a CPU counterfactual (reuse the `--ts-shadow` harness) that does
   localized reopt (only branches within distance d of applied moves) each round + one final full reopt, vs
   the full-reopt baseline; compare final lnL + RF. Cheap: one CPU run, no GPU.
 - **GO** to a GPU LBR build (a stale-mask + per-round affected-edge list threaded into `gpu_jolt_optimize`'s
   sweeps) only if **G1 (`af`≤0.30) AND G2 (final lnL within ε, RF=0)** both hold. Else FALSIFIED — bank the
   finding, the GPU per-search path is already near its localizable limit.

---
*Status: hypothesis authored + self-red-teamed (the naive O(log n) version refuted). Next: Plan-agent to design
the af/exactness measurement, implement, run on AA-100K, then independent red-team — then GO/NO-GO.*
