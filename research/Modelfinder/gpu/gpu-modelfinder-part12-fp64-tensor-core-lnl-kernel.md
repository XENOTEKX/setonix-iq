# Part 12 — FP64 tensor-core MMA for the JOLT lnL/gradient kernel (a *conditional* lever, kill-switch first)

**Status: CLOSED — T.0 kill-switch fired STOP (2026-06-18).** The standalone decider measured the FP64 tensor-core
(wmma-double DMMA) matvec at **0.36–0.60× the speed of JOLT's scalar matvec** on H200 *and* A100 (i.e. 1.6–2.8× SLOWER),
parity bit-identical. Gate was ≥1.3× at 1M on H200; got 0.36×. **Lever closed — see §XII.6.** This is the "Acceptable-NO"
case the plan was built to catch: it cost 0.43 SU / ~10 s of GPU instead of the 6–10 days of in-tree T.1–T.3 work, and it
*explains* why JOLT-on-CUDA-cores already ties tensor-core BEAGLE. The plan below (§XII.1–5) is retained as the rationale.

**Status (history):** PLAN (researched 2026-06-18, author as1708; two independent read-only agents + BEAGLE-4.0 reference study).
**Origin:** the JOLT-vs-BEAGLE-4.0 head-to-head (CHANGELOG §B, jobs 171265226/171267592/171269929/171270072, commit
`960121f8` local). **Companion:** part8 (JOLT code audit / perf levers), part11 (commercial-card precision), and the
`ref_beagle4_tensorcore_2026` note. **Code:** `tree/gpu/gpu_lnl_intree.cu` (kernels), `tree/phylotreegpu.cpp` (driver).

---

## §XII.0 — The honest reframe (READ THIS FIRST)

The benchmark line that motivated this — *"H200 lnL: BEAGLE tensor-core 15.27 ms vs JOLT 17.51 ms (~13% behind) ⇒
adopt FP64 tensor cores in `k1_node`"* — is **true but oversells the lever.** Two independent analyses of the actual
JOLT kernels and BEAGLE's reference tensor-core source converge on a more cautious verdict:

> **JOLT's *CUDA-core* eigenspace kernel already ties BEAGLE's *tensor-core* kernel** (A100 dead heat 26.51 vs 26.50;
> H200 only 13% behind). It does so **precisely because** JOLT is register-resident and fused, while BEAGLE's tensor
> path pays the overheads tensor cores force: shared-memory staging of every partial tile, padding the 20-state
> matrix up to **32** (≈37% of the MMA tile is zero), a bank-conflict swizzle, and a layout transpose. **JOLT's
> scalar approach already captured most of the gain tensor cores would otherwise buy** — by avoiding all of that.

So this is **not** a slam-dunk "free 2×". It is a **modest, conditional** lever whose ceiling is ~2× on the GEMM
portion *only in a compute-bound regime*, and JOLT's kernels are **memory-latency / scheduler-starvation-bound**, NOT
FLOP-bound (00-MASTER §5: `k1_node`/`kj_pre` 56 regs / ~57% occupancy; the production `kj_derv_fused` is 32 regs /
**100% occupancy** on A100/H200 yet still latency-bound — `issue_active ~8%`, `math_pipe 0.08%`). A FLOP-doubling tensor
path does little when the SMs are stalled on dependent-kernel latency and memory, not arithmetic. The realistic outcomes:

| Regime | Honest expectation |
|---|---|
| AA-100K | **likely a wash** (latency-bound; same lesson as the K4 kernel-fusion wash-to-loss at 100K — 00-MASTER §3, part5) |
| AA-1M / 10M | **possible 1.3–2× — but see the padding break-even (§XII.5): the 20→32 pad can erase it.** Only regime where the FLOP ratio could show |
| DNA (4-state) | **expected loss** (inferred from kernel structure, *not* measured) — a 4×4 padded into 8×8×4 wastes ≥75% of the tile; keep scalar |
| V100 (sm_70) | **impossible** — Volta has no FP64 tensor cores; the instruction is illegal silicon there |

Therefore the plan is **kill-switch-first** (T.0 decides go/no-go *before* any in-tree change), mirroring G.3.0 /
G.4.0b. If T.0 shows the register-resident scalar kernel already wins, **we close this lever and say so** — that is
itself a publishable, honest result (it explains *why* a hand-rolled FP64 CUDA kernel matched tensor-core BEAGLE).

---

## §XII.1 — Why the lever is conditional (the kernel-level facts)

### What the JOLT kernels actually compute (agent A, `gpu_lnl_intree.cu`)
- The diagonal `exp(λt)` scaling is **pre-baked host-side into `echild`** (`echild = U·exp(eval·rate·len)`, line ~16),
  so the per-node work is **~97% GEMM-shaped matvec** at ns=20: `accum_child` (lines 45–59) is an `ns×ns · ns`-vector
  product per child (`v += ecc[x*ns+i]*pc[i*nptn]`), the back-transform (line ~82) `o[r]=Σ_x g_Uinv[r*ns+x]·prod[x]` is
  another, and the child Hadamard product (`prod[x]*=v`) is the only non-GEMM piece (≈3% of FLOPs).
- `kj_pre` (gradient preorder, lines ~178–186) is the same shape: ~97% GEMM-able `U`/`U⁻¹` matvecs, ~3% diagonal+Hadamard.

**So the matvec IS GEMM-able — but the shape defeats tensor cores three ways:**
1. **M = K = ns = 20** (4 for DNA). A DMMA fragment is m8n8k4; 20 must pad to **32** (BEAGLE's `kPaddedStateCount`),
   so the 2-D tile waste is **≈61% (1 − 20²/32² = 1 − 400/1024)** — i.e. the DMMA does ~2.56× the FLOPs to produce the
   20×20 result. Only N (patterns) is "long." Tensor cores cannot turn a 20×20 operand into throughput.
2. **Latency-bound, not FLOP-bound.** The 2×-peak FP64-tensor-vs-CUDA-core ratio is only realized when compute-bound.
   The production AA path is **memory-latency / scheduler-starvation-bound** (00-MASTER §5; not low occupancy — the
   production fused kernel is 100% occ), so a FLOP-doubling path sees *well under* 2× — plausibly a wash at 100K.
3. **Layout fights the fragment.** Partials are **state-major / pattern-minor** (`offset = (c*ns+r)*nptn + ptn`,
   stride-1 axis = ptn). Feeding a DMMA fragment needs an `ns×N_tile` block staged to shared memory with a transpose;
   that staging traffic hits the *exact* resource (L1/shared/latency) that is already the bottleneck. The current
   register-resident inline matvec avoids it entirely.

### Determinism is a hard constraint (agent A)
The cross-pattern lnL reduction is **deterministic by design**: `kj_reduce3` (lines ~204–219) is a shared-memory tree
reduce (no `atomicAdd`), host-combined with Kahan in fixed order — "bit-reproducible across launches so the LM
accept/reject trajectory is stable" (G.5.0). **The reduction must NOT be routed through DMMA** (it would change
summation order and destabilize the optimizer trajectory). DMMA is confined to the *matvec*, never the reduction.

### Parity drops from bit-identical to rel ≤ 1e-12 (agents A+B)
FP64 tensor cores are **true IEEE double** (no TF32/FP16 truncation — compatible with the non-negotiable FP64 rule),
**but the multiply-accumulate order differs** from the scalar FMA loop. Result: agreement to ~machine-eps per node,
~1e-13…1e-12 after a deep peeling tree — **never `rel 0.0`**. The project's scalar kernels hit bit-identical because
they replicate the exact FMA order; a tensor kernel structurally cannot. **The tensor-path parity gate is therefore
`rel ≤ 1e-12`** (same class as the FD-gradient gates and G.7.1's 4.465e-13), with the **scalar kernel retained as the
bit-identical oracle**. Do not regress to "bit-identical" as the acceptance criterion for the tensor path.

---

## §XII.2 — The reference implementation (BEAGLE-4.0 tensor-cores, agent B)

Source `/scratch/rc29/as1708/beagle-tensorcores/libhmsbeagle/GPU/kernels/` (branch `tensor-cores @ dd962d48`, the exact
Gangavarapu 2026 code). The proven recipe for 20-state:
- **Raw inline PTX**, not `wmma`/CUTLASS/cuBLAS: `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64`
  (`kernelsXTensorCore.cu:117-119, 124-126`; 208 instances across state counts in `BeagleTensorCore_kernels.h`, all
  `.target sm_80`). `#include <mma.h>` appears (in `kernelsX.cu:19`, not the tensor file) but is **vestigial** — no
  `nvcuda::wmma` is actually used anywhere; the MMA is hand-written PTX.
- **Pad 20 → 32** (`BeagleGPUImpl.hpp:321-322`); 20 is never compiled directly — AA runs as the 32-state kernel.
- **8 patterns/block**, split 4+4 (`PATTERN_SPAN`); warp/lane fragment math at lines 39–44; K walked in steps of 4
  (8 iters for 32 states), tiling loop 95–129; **bank-conflict swizzle** macros `GET_SMEM_OFFSET_*` (lines 60–74).
- **Gamma categories ride OUTSIDE the MMA**: each category is a separate transition matrix dispatched over the grid
  (partials kernel) or a serial per-category loop with post-MMA weight (`kernelsXDerivativesTensorCore.cu:225`). The
  tensor core only ever sees a plain 20(→32)-state matrix×partials product.
- **The peeling product is post-MMA**: two MMA accumulators (left child, sibling) are multiplied element-wise
  (`res11*res21`, `kernelsXTensorCore.cu:137`) to form the parent partial — exactly JOLT's `prod[x]*=v` Hadamard, after DMMA.
- **Selection/arch gate**: `BEAGLE_FLAG_VECTOR_TENSOR` + `GetSupportsDoublePrecision` (`TensorCoresPlugin.cpp:48-53`);
  peeling block.y forced to 4 (`KernelLauncher.cpp:1061-1077`).
- **DNA (`kernels4TensorCore.cu`)** shows the contortions (zero-out half the A operand, `laneId<16` write guard,
  lines 57–88) that confirm 4-state is not worth it.

**API choice for us:** prototype correctness with `nvcuda::wmma` double fragments (less lane math), ship with raw
`mma.sync.m8n8k4.f64` PTX (lets us fuse the matvec + Hadamard + gamma in registers/shared, like BEAGLE). cuBLASLt /
CUTLASS are wrong here (tiny M=K, and they can't fuse the peeling product or the tree traversal).

---

## §XII.3 — Phased plan (kill-switch first)

### T.0 — KILL-SWITCH DECIDER (standalone, A100 **and** H200) — *gates everything*
Build a standalone micro-benchmark (new `tree/gpu/tc_decider.cu`, nvcc, no in-tree change) that runs the **20→32 padded
FP64 DMMA matvec batched over N patterns** and compares against JOLT's *current* register-resident scalar matvec at
**matched dims N = {100K, 1M}**. (All testing on A100/H200 — V100 has no FP64 TC, so it cannot even compile/run the path;
the V100 stays the scalar oracle. This moves the correctness loop off the usual `gpuvolta` dev queue onto `dgxa100`/`gpuhopper`.)
- **Correctness:** DMMA result vs scalar oracle, gate **rel ≤ 1e-12** (not bit-identical — §XII.1).
- **Wall:** ms/eval, DMMA vs scalar, on A100 (`dgxa100`) and H200 (`gpuhopper`).
- **The 100K arm is a THESIS TEST, not just a warm-up.** §XII.0 asserts JOLT's fused scalar kernel "already captured the
  tensor gain," which is *why* tensor-core BEAGLE only tied JOLT at 100K while beating its own un-fused CUDA kernel 2.43×.
  T.0's 100K result decides it: if DMMA **beats** JOLT-scalar at 100K, the thesis is **falsified** and the lever is bigger
  than claimed; if it's a **wash**, the thesis is **confirmed** (and 1M becomes the only hope). Report both, don't assume.
- **Prototype** with `wmma` double for speed-to-correctness, then raw PTX for the real timing.

> **GATE (deploy-card-specific, not "either"):** proceed **only if** DMMA is **≥ ~1.3× faster than scalar at N = 1M with
> rel ≤ 1e-12 on the card we deploy at scale — H200 (the 1M/10M target, part7).** If it wins *only* on A100 (where the
> benchmark was already a dead heat, i.e. weak evidence) but not H200, that does **not** justify the permanent two-path
> maintenance cost (§XII.5) — treat as STOP. If it is a wash-or-loss (the honest prior, given JOLT already ties
> tensor-core BEAGLE), **STOP and document**: "the register-resident eigenspace scalar kernel already captures the
> tensor-core gain; FP64 TC is not a lever for this kernel shape." That closes the lever cleanly and explains the tie.

### T.1 — gradient kernel FIRST (`kj_pre` / `kj_derv_fused`) — the wall-dominant target (only if T.0 passes)
**The gradient dominates the JOLT wall ~4.3:1** (CHANGELOG §B, H200: all-branch gradient 74.54 ms vs lnL 17.51 ms), so an
lnL-only speedup barely moves the end-to-end CTF wall — therefore the gradient kernel is the *first* in-tree target, not
the second. Add a DMMA path for `kj_pre`'s `U`/`U⁻¹` matvecs (lines ~184–186) behind **`#if __CUDA_ARCH__ >= 800`** + a
runtime device-capability check + an opt-in **`JOLT_TENSOR=1`** env flag (default OFF). Stage `ns×N_tile` to shared
memory with a BEAGLE-style swizzle; keep the Hadamard/diagonal pieces and the reduction scalar (`kj_reduce3`). **FD-validate
the gradient** (df ~1e-9 / ddf ~1e-6, 00-MASTER §4 #8 discipline) **and** parity rel ≤ 1e-12 before any gate; **CPU path
byte-identical**; engage only for `ns==20`.

### T.2 — lnL kernel `k1_node` (only if T.1 shows a real end-to-end win)
Apply the same DMMA path to `accum_child`'s matvec. Lower priority precisely because it is ~1/4.3 of the wall — do it only
if T.1 demonstrates the gradient win is real end-to-end (else the lnL change is maintenance cost for ~0 benefit). Gate:
lnL rel ≤ 1e-12 vs scalar; reduction stays scalar; `ns==20` only.

### T.3 — end-to-end CTF wall + energy (A100/H200, AA-1M/10M)
The decisive test: does `JOLT_TENSOR=1` move the **CTF wall** (where lnL+gradient sweeps × 27 iters dominate) at 1M/10M?
Report as a **curve vs pattern count** (never one headline number — 00-MASTER §4 #11), with energy (Wh) alongside. Route
100K, DNA, and V100 to the scalar path. Compare against the frozen parity binary's published CTF rows.

---

## §XII.4 — Scope guards (what NOT to do)
1. **AA (ns=20) only.** DNA 4-state is an *expected* loss (inferred from the kernel structure, not benchmarked); keep
   scalar. AA +FO/GTR20 unaffected (orthogonal — eligibility).
2. **A100/H200 only.** Compile-gate on `__CUDA_ARCH__>=800` + runtime check; V100 stays scalar (also the bit-identical oracle).
3. **Never TF32/FP16/fast-math.** FP64 DMMA only (true IEEE double). The reduced lnL/gradient sums stay FP64 (00-MASTER §4 #7).
4. **Never route the reduction through DMMA** — it breaks the deterministic, bit-reproducible LM trajectory (§XII.1).
5. **Scalar kernel stays the default and the oracle;** tensor path is opt-in (`JOLT_TENSOR`), parity-gated rel ≤ 1e-12.
6. **No claim of a win before T.0 (decider) and T.3 (end-to-end) measure it.** The honest prior is "wash at 100K,
   maybe 1.3–2× at 1M/10M" — and a clean NO at T.0 is an acceptable, useful outcome.

## §XII.5 — Honest expected payoff (and the break-even that could kill it)

**The padding break-even (the crux T.0 must resolve).** DMMA peak FP64 ≈ 2× CUDA-core FP64. But the 20→32 pad means the
DMMA does ~2.56× the FLOPs (1024/400) to produce the 20×20 result. Naive FLOP-time: `2.56 / 2 ≈ 1.28×` the GEMM time of a
*perfectly-packed* kernel — i.e. **the padding alone can erase the 2× tensor advantage and leave DMMA slower than a tight
scalar matvec.** BEAGLE still won 2.43× over *its own* CUDA path only because that path was un-fused and memory-heavy —
tensor cores recovered BEAGLE's slack up to JOLT's already-fused level. **JOLT's scalar baseline is already register-fused,
so it has far less slack for DMMA to recover.** Plus DMMA forces the partials out of registers into shared memory (the
layout transpose, §XII.1), adding traffic to the bound resource. Net prior: **wash-to-loss at 100K, marginal at 1M** — and
a clean NO is a real possibility, not a failure.

- **Best case:** ~1.3–2× on the AA-1M/10M gradient+lnL sweeps on H200 ⇒ a meaningful but not transformational cut to the
  CTF refine wall in the large-N regime (the one regime where JOLT is already the clean win — part5/part7).
- **Likely case:** wash at 100K; small-or-no win at 1M; net a modest CTF-wall improvement on big alignments only.
- **Acceptable-NO case:** T.0 confirms the register-resident scalar kernel already wins ⇒ close the lever, and the
  benchmark tie (JOLT-CUDA ≈ BEAGLE-tensor) is *explained* rather than chased. Either way the result is honest and citable.
- **Carrying cost (weigh against the win).** Shipping the tensor path means **two kernel code paths forever** (scalar
  oracle + DMMA), gated by arch + env, validated on **2+ architectures** — every future kernel change must be re-proven on
  both, on A100/H200 queues. For a *conditional, big-N-only, ~1.3–2×* lever this maintenance + test-matrix doubling is
  non-trivial and could be **net-negative**; T.3's measured end-to-end gain must clear it, not just the T.0 micro-bench.
- **Effort:** ~6–10 days to the T.0 decider (the gate); T.1–T.3 only if T.0 passes on H200.

---

## §XII.6 — T.0 RESULT (2026-06-18) — the kill-switch fired: **STOP**

**Harness:** `tree/gpu/tc_decider.cu` (nvcc, sm_80+sm_90 fat binary), runner `gadi-ci/gpu-modelfinder/run_tc_decider.sh`.
Agent-reviewed before submit (wmma fragment mapping verified element-by-element vs CUDA-12.5 `mma.hpp` — no transpose bug;
scalar baseline = faithful copy of `accum_child:50-51`; DMMA stages M to shared like BEAGLE; parity over real states only).
**Jobs 171587052 (H200/gpuhopper) + 171587053 (A100/dgxa100), both exit 0.**

| Card | nptn | scalar | DMMA | DMMA speedup (scalar/dmma) | parity vs scalar |
|---|---|---|---|---|---|
| **H200** | 100K | 0.0725 ms | 0.1291 ms | **0.56×** (1.8× slower) | **rel 0.0** (bit-identical) |
| **H200** | **1M (gate)** | 0.4170 ms | 1.1666 ms | **0.36×** (2.8× slower) | **rel 0.0** |
| A100 | 100K | 0.1526 ms | 0.2652 ms | 0.59× | rel 0.0 |
| A100 | 1M | 1.2268 ms | 2.0255 ms | 0.60× | rel 0.0 |

**Verdict: STOP.** Gate was ≥1.3× at 1M on H200; measured **0.36×**. The FP64 tensor-core matvec is **1.6–2.8× SLOWER**
than JOLT's existing register-resident scalar matvec, on both cards, every dimension. **Parity was bit-identical** (rel 0.0) —
the §XII.1 "rel ≤ 1e-12 not bit-identical" concession turned out unneeded for this matvec (wmma double's accumulation
matched the scalar order exactly), so correctness was never the blocker; **speed is.**

**Why (confirms §XII.0 thesis):** the 20→32 pad makes DMMA do ~2.56× the FLOPs; the matvec is latency/bandwidth-bound not
FLOP-bound (so the ~2× FP64-TC peak ratio is never realized); and DMMA pays shared-staging + strided fragment loads that
JOLT's register-resident inline matvec avoids. The eigenspace+register approach already captured the gain.

**Honest caveat + why it doesn't change the verdict:** this is the *naive* `wmma` prototype (B loaded unstaged from global).
A fully-tuned raw-PTX version (shared-staged + swizzled B, per BEAGLE) would beat this prototype — **but** the independent
ceiling is already known: BEAGLE's *fully-tuned* tensor kernel only beats JOLT-scalar by **~13% on H200** (lnL 15.27 vs
17.51 ms, CHANGELOG §B) — **below the 1.3× gate**. Both the naive decider (0.36×) and BEAGLE's tuned kernel (1.15×)
converge: the lever does not clear the bar. Building the tuned PTX version would spend SU/days to reconfirm a known NO.

**Citable conclusion:** *A hand-rolled FP64 CUDA-core eigenspace matvec, register-resident and fused, beats an FP64
tensor-core (DMMA) matvec for the 20-state phylogenetic-likelihood kernel — because the operation is latency-bound and the
20→32 padding negates the tensor-core FLOP advantage. This is why JOLT on ordinary CUDA cores ties tensor-core BEAGLE.*
