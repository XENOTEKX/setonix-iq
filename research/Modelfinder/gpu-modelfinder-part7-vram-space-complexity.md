# PART VII — VRAM space-complexity reduction: making per-model JOLT fit commercial GPUs (deep-research item)

**Author:** as1708 / Claude Fable 5 (xhigh), 2026-06-11. Opened at the user's direction after G.4.3b (+I in JOLT)
validated: *"most of IQ-TREE's users will not have access to A100/H100 — most use commercial GPUs with less VRAM,
so space-complexity reduction is necessary... an important feature we will need to implement."* **Status: SCOPED +
first-principles analysis + recommended direction. The build is FUTURE WORK (not started).**

---

## VII.0 The problem, in one paragraph

JOLT's per-model GPU memory is **O(nInternal · nptn)** because the postorder partial-likelihood arena keeps **one
slot per internal node** (`gbj_partial = nInternal · ncat · ns · nptn · 8` bytes) and is **not recycled**. At
AA-1M this is **~88.6 GB per model** (measured-from-code breakdown below) — it fits only **H200 (141 GB)**, OOMs
**A100 (80 GB)**, and is far beyond **V100 (32 GB)** or consumer cards (RTX 4090 24 GB, typical 8–16 GB). Since the
JOLT mutex serializes the GPU anyway (one model at a time), the binding constraint for accessibility is **per-model
footprint**, not aggregate. The fix that is **exact, tunable, and parity-preserving is PATTERN TILING**: process the
alignment in pattern-chunks of size `nptn/T`, shrinking every partial buffer by T while keeping the likelihood and
all gradients bit-exact (they are sums over patterns). Tiling at T=10 brings AA-1M to **8.9 GB (fits V100/RTX4090)**;
T=40 to **2.2 GB (fits any GPU)**.

---

## VII.1 Measured memory breakdown (from `gpu_jolt_optimize` allocations, `tree/gpu/gpu_lnl_intree.cu`)

`slotSz = ncat·ns·nptn·8` bytes (one [cat][state][pattern] partial). ns=20, ncat=4, 100 taxa ⇒ nInternal=98,
tree height ≈44 ⇒ preorder pool nPool≈46.

| buffer | sizing | AA-100K (96K ptn) | AA-1M (940K ptn) | AA-10M (9.4M ptn) |
|---|---|---:|---:|---:|
| **postorder partials** `gbj_partial` | **nInternal·slotSz** | **6.0 GB** | **59.0 GB** | **589.6 GB** |
| preorder pool `gbj_prepool` | nPool·slotSz (already O(depth)) | 2.8 GB | 27.7 GB | 276.7 GB |
| scratch (theta/pretmp/tipeig) | 3·slotSz | 0.2 GB | 1.8 GB | 18.0 GB |
| small (patlh/pdf/pddf/rnum/tip/echild/baseinvar) | O(nptn)+O(nnodes) | ~0.2 GB | ~0.2 GB | ~1.6 GB |
| **TOTAL per model** | | **~9.0 GB** | **~88.6 GB** | **~886 GB** |

**Two facts the table makes obvious:**
1. **The postorder arena (59 GB at 1M) dominates** and is the one buffer NOT recycled — every internal node owns a
   distinct slot for the whole sweep.
2. The preorder pool is **already O(depth)** (G.4.0b recycling: ~46 live slots, not 98) — yet at 1M it is still
   27.7 GB because each slot is 0.6 GB. **The slot size itself (∝ nptn) is the real lever**, which is exactly what
   tiling attacks.

**Target hardware (per-model footprint must fit ONE of these for accessibility):**
`V100 32 · RTX 4090 24 · A100 80 · H200 141 · typical consumer 8–16` GB.

---

## VII.2 Why O(depth) postorder recycling alone is NOT enough (and is hard for the gradient)

The obvious idea — recycle the postorder arena to ~tree-height slots (44 vs 98), as the lnL-only G.4.0b sweep does
— **does not transfer cleanly to the JOLT gradient and does not get small enough anyway:**

- **It does not transfer:** the Ji-2020 linear-time gradient needs a **preorder pass** (`kj_pre` computes `pre_v` =
  "rest of tree above edge u→v"), and that pass reads the **postorder partials of v's siblings/subtrees**. So the
  postorder partials must stay **live through the preorder sweep** — they cannot be freed bottom-up the way an
  lnL-only postorder can. A fused post+pre subtree schedule (compute `pre_v` for a subtree immediately after its
  postorder, freeing as you ascend) is possible but **intricate and easy to get wrong** (the very class of subtle
  bug the G.4.3b rate-scaling miss and the +R+I gate miss just illustrated).
- **It is insufficient at scale:** even recycled to 44 slots, AA-1M is 44·0.6 = **26 GB** (postorder) + 27.7 GB
  (preorder) ≈ **54 GB** — still OOMs A100-80? No (fits A100), but **still far over V100-32 and consumer cards.**
  Recycling buys ~1.6×; it does **not** reach commercial GPUs at 1M.

**Conclusion:** recycling is a secondary optimization. The slot SIZE (∝ nptn) is the dominant term, so the primary
lever must shrink nptn-per-buffer. That is tiling.

---

## VII.3 The recommended direction — PATTERN TILING (exact, tunable, parity-preserving)

**Why it is exact.** Every quantity JOLT needs is a **sum over patterns**:
`lnL = Σ_p w_p·log L_p`, `df_e = Σ_p w_p·(∂log L_p/∂b_e)`, `ddf_e = Σ_p w_p·(…)`, `gradR_c = Σ_p w_p·rnum_{c,p}·invL_p`,
`gradPinv = (Σ_chunks lnL(pinv+ε) − Σ_chunks lnL)/ε`. The per-pattern partials `partial[c][x][p]` depend ONLY on
pattern p's tip column (plus the shared tree/eigen/rates). So partitioning patterns into T chunks `[p0,p1)` and
running a full postorder+preorder sweep **per chunk**, then **accumulating** each chunk's contribution to the
reductions, reproduces the one-shot result to the **last bit** (Kahan-summed across chunks). No approximation — this
is the same additivity that already underlies the ptn_freq-weighted reductions.

**Why it shrinks memory by exactly T.** Every partial buffer is sized `ncat·ns·CHUNK·8` with `CHUNK = nptn/T`. So
`gbj_partial`, `gbj_prepool`, scratch, patlh/pdf/pddf/rnum all shrink by T. Footprint ≈ **88.6 GB / T** at 1M:

| T (chunks) | AA-1M /model | AA-10M /model | fits |
|---:|---:|---:|---|
| 1 | 88.6 GB | 886 GB | H200 only / none |
| 4 | 22.2 GB | 221 GB | A100, RTX4090(1M), V100? (close) / none |
| 10 | **8.9 GB** | 88.6 GB | **V100, RTX4090, all ≥12 GB** / H200 |
| 40 | **2.2 GB** | 22.2 GB | **any GPU incl. 8 GB consumer** / A100 |

**Cost (the time-space tradeoff).** Each `evalLnL`/`computeGradient` becomes T sweeps instead of 1: ~T× more kernel
launches and T× re-upload of the (small, ~2.5 MB) echild/expfac constants per chunk. The per-chunk work at 1M/T=10
is a ~94K-pattern sweep — i.e. the *already-efficient 100K regime*. Per the K3 (CUDA-graph parity) and K4
(fusion-wash) findings, launch latency at this size is largely hidden, so the expected overhead is **modest
(launch + constant-reload bound), not T×**. **This must be measured** — tiling has only ever been costed as a
*throughput* lever (part5 P3.1, cancelled) and as a *capability* enabler (this doc); the per-model wall hit of
T-way tiling on JOLT is unmeasured.

**Implementation sketch (moderate, well-contained):**
- Add `int nTile` (or auto-pick from a target VRAM budget and `cudaMemGetInfo`). Outer loop over pattern chunks.
- All partial buffers allocate at `CHUNK` width; `tip`, `ptn_freq`, `base_invar` index into the chunk.
- Postorder + preorder + `kj_derv`/`kj_ratenum` run per chunk; **host-side Kahan accumulators** for lnL, df[e],
  ddf[e], rnum (or accumulate rnum on-device across chunks before the gradR reduction). The branch/alpha/pinv LM
  step is **unchanged** — it consumes the fully-accumulated gradients.
- The echild/expfac/eigen constants are chunk-independent ⇒ build once, reuse across chunks (only the partials are
  chunk-local). `g_val0/1/2`, `g_rscale` are chunk-independent too.
- **Gate:** chunked lnL == one-shot lnL to machine precision (rel ≤ 1e-12), and the JOLT MLE unchanged; then the
  per-model wall vs T (the overhead measurement).

**This is the single highest-value VRAM item: it is exact, it is tunable to ANY GPU, and it reuses every existing
kernel unchanged (only the host orchestration changes).** It also subsumes the AA-10M *capability* frontier (10M at
T=40 = 22 GB fits a V100; one-shot 886 GB fits nothing).

---

## VII.4 Secondary directions (research, lower priority)

1. **O(depth) postorder recycling (fused post+pre traversal).** Compose with tiling for a further ~1.6× (98→44
   slots). Worth it only after tiling lands and only if a target GPU sits just above the tiled footprint. Risk:
   the fused schedule's correctness (the preorder-needs-postorder-live constraint, §VII.2).

2. **Mixed-precision STORAGE (FP32 partials, FP64 reductions).** Halves every buffer (88.6→44 GB at 1M). **Risk:
   parity** — FP32 partials carry ~7 digits, so the reduced lnL/gradient parity would fall from ~1e-12 toward
   ~1e-6, violating the standing FP64-parity constraint for the *exact-refine* path. **Candidate use: the COARSE
   ranking phase only** (where BIC ranking tolerates ~1e-6), NOT the fine refine. Could combine with tiling
   (FP32 + T-tiling = 2T× reduction). Needs an error-analysis + a parity gate before trust.

3. **Unified memory / out-of-core (spill partials to host RAM).** `cudaMallocManaged` + prefetch, or explicit
   host-staging of cold partials. **Likely bandwidth-bound** (PCIe ~16–64 GB/s « HBM), so probably slow; a
   last-resort capability enabler, not a performance path. Tiling is strictly better when applicable (it never
   leaves VRAM).

4. **Recomputation / gradient checkpointing.** Store only O(√n) or O(depth) postorder partials, recompute the rest
   during the preorder. Classic time-space tradeoff; more recompute than tiling for similar memory, and more
   complex. Tiling dominates it for this workload (tiling's "recompute" is just the next chunk, with perfect
   locality).

---

## VII.5 Recommended phased plan (when this is picked up)

| Phase | Deliverable | Gate |
|---|---|---|
| **V.A — tiling correctness** | Outer pattern-chunk loop in `gpu_jolt_optimize`; per-chunk sweep + Kahan-accumulated lnL/df/ddf/rnum; auto `nTile` from `cudaMemGetInfo` vs a budget. | chunked lnL == one-shot rel ≤ 1e-12 on AA-100K (V100); JOLT MLE bit-unchanged; +I/+G/+F all still pass the G.4.3b gates. |
| **V.B — tiling overhead** | Per-model JOLT wall vs T∈{1,4,10,40} at AA-1M on V100 (T=10 should fit 32 GB). | wall(T=10)/wall(T=1, on H200) overhead characterized; report the launch+reload cost. |
| **V.C — capability frontier** | AA-1M `--jolt` refine running on a **V100** (T≈10) and ideally a consumer card; AA-10M on a V100 (T≈40). | correct LG+(I+)G4 MLE produced on 32 GB / consumer VRAM — the accessibility headline. |
| **V.D (optional)** | FP32-storage coarse phase + tiling, with an error gate; and/or fused post+pre recycling. | coarse BIC ranking unchanged with FP32 partials; recycling correctness vs one-shot. |

---

## VII.6 Why this matters (the accessibility thesis)

The whole G.4 line (JOLT + CTF) is positioned as a **single-device tool speedup for users without a cluster**
(part5 V.5, part6). That positioning is **hollow if the single device must be an H200** — the users who lack a CPU
cluster also lack an H200; they have a workstation RTX card or a single V100. **Pattern tiling is what makes the
"runs on your GPU" claim true for the people the claim is for.** It converts JOLT from "fits a 141 GB datacenter
GPU at 1M" into "fits any GPU at any data size, by choosing T" — at a modest, measurable wall cost. It is therefore
not a nice-to-have but a **prerequisite for the accessibility story** the project rests on.

**Bottom line:** per-model footprint is O(nInternal·nptn) ≈ 88.6 GB at AA-1M; the postorder arena dominates;
**pattern tiling (exact, tunable, kernel-reuse) is the recommended fix** and reaches V100/consumer GPUs at any
scale; recycling/mixed-precision/out-of-core are secondary. Build is future work; correctness gate first (chunked
== one-shot rel ≤ 1e-12), overhead measurement second.
