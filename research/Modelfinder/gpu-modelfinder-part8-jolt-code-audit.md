# PART VIII — Honest code audit of the JOLT GPU optimizer (correctness + performance + simplification)

**Author:** as1708 / Claude Fable 5 (xhigh), 2026-06-11. A full, deliberately self-critical audit of the code we
built across G.4.0→G.4.3b, after +I (pinv) was added and validated. Two independent passes: (A) a correctness +
edge-case sub-agent (`adeb1336`, 2026-06-11) and (B) a performance + simplification sub-agent (`a7f213f1`, Fable),
plus a line-by-line read by the author. Files audited: `tree/gpu/gpu_lnl_intree.cu`, `tree/phylotreegpu.cpp`.
**Honesty note:** wall-cost figures below are *code-derived estimates* (bytes moved, op counts), NOT measured —
they are tagged ⏳; the AA-1M H200 run (job 170517590) is the first measurement that will confirm or rerank them.

---

## VIII.0 Verdict in one paragraph

**The code is CORRECT** — empirically GPU≡CPU to rel 1.7e-12 at pinv up to 0.50 (the 2× rate-scaling stress),
parity vs IQ-TREE's own MLE, and the one real correctness bug found in audit (the +R+I gate, §VIII.1) is fixed.
**The code is NOT yet fast at scale.** The single biggest avoidable cost is **`reduceDerv`: a host-side,
per-pattern, per-edge Device→Host reduction** that moves ~4.4 GB and runs a 555M-iteration single-threaded Kahan
loop **per gradient sweep at 1M** (≈35–40% of the sweep, ⏳). Moving that reduction on-device is the #1 lever, and
three more (skip a redundant base sweep; fuse theta/derv/ratenum; the pinv finite-difference sweep) compound it.
**The code is correct but un-optimized; none of the optimizations are needed for correctness, and whether any are
needed to beat 2/4 CPU nodes at 1M depends on the H200 wall (pending).** Separately, per-model memory is ~85–88 GB
at 1M (the postorder arena) — a *scalability ceiling* addressed by pattern tiling (PART VII), independent of wall.

---

## VIII.1 Correctness (PASS — with one bug found and fixed)

- **+I (pinv) math validated at all pinv.** `L_p = lh + pinv·base_invar[p]`; `base_invar` replicates
  `PhyloTree::computePtnInvar` exactly (DNA/protein ambiguous, STATE_UNKNOWN, +F empirical freqs all checked);
  IQ-TREE's `RateGammaInvar` rate scaling `meanR/(1−pinv)` is matched; pinv gradient by finite difference is robust
  to the rate↔prop↔pinv coupling. Self-check GPU≡CPU: **rel 1.77e-12 (pinv 0.0013), 1.69e-12 (pinv 0.50).**
- **BUG FOUND + FIXED — the +R+I eligibility gate.** The gate discriminated gamma-vs-freerate by
  `getGammaShape() ≤ 0`, but `RateFree` (+R) inherits a *positive* `gamma_shape`, so **+R / +R+I would have wrongly
  engaged JOLT** (uniform proportions + mean-gamma rates) and — because writeback precedes the self-check —
  returned a silently-corrupt result on a `-m MFP` run (it does NOT bite `-m TESTONLY`, which has no +R). Fixed by
  the robust `site_rate->isGammaRate() == GAMMA_CUT_MEAN` discriminator (declines +R, +R+I, and median-gamma `+Gm`
  at once), plus declining `-no_rescale_gamma_invar` and a backward-FD step at the pinv upper boundary.
- **No other correctness defects** in: base_invar frequency source, rate/prop writeback order
  (`setPInvar`+`setGammaShape`+`clearAllPartialLH`), division-by-(1−pinv) safety (bounded by `frac_const_sites<1`),
  the non-+I path (byte-identical when `optPinv==0`), or mixture/site-specific/fixed-pinv declines.

---

## VIII.2 Performance findings (ranked by ⏳ estimated AA-1M payoff)

Constants at 1M: nptn≈940K, ncat=4, ns=20, nInternal≈98, nedge≈197; `slotSz=ncat·ns·nptn`=**601 MB/slot**;
one nptn D2H = **7.5 MB**. The LM loop runs ~13–92 iterations; each = 1 `computeGradient` (walks ~197 edges) +
up to ~15 `evalLnL`.

### #1 — `reduceDerv` host round-trip (CONFIRMED, do first). `gpu_lnl_intree.cu:484`
Three blocking D2H (`patlh`,`pdf`,`pddf`) + a single-threaded triple-Kahan host loop, called **once per edge**
(~197×/gradient sweep) plus once per `evalLnL`. ⏳ **~4.4 GB D2H + 555M serial host iterations per gradient sweep
≈ 0.95 s (~35–40% of the sweep).** Scales as **nptn×nedge×iters — the worst scaling in the file.** Two compounding
wastes: `proc` uses the returned lnL only on the *first* edge yet D2H's `patlh` on all 197 (≈1.5 GB/sweep
discarded); `evalLnL` uses only lnL yet D2H's `pdf`/`pddf` too.
**Fix:** do the `ptn_freq`-weighted reduction **on-device** (a reduction kernel / block-reduce appended to
`kj_derv`, returning 3 scalars per edge; upload `ptn_freq` once H2D). D2H drops to 24 bytes/edge.
**Payoff ⏳ ~1.6× on the gradient sweep. Risk:** low–med (FP64 tree-reduction reorders summation vs host Kahan at
~1e-13 — re-validate the 1e-9/1e-6 parity gates).

### #2 — Redundant base-point sweep in `computeGradient` (CONFIRMED). `gpu_lnl_intree.cu:511`
`computeGradient` always does `rebuildEchild(); postorderFill();` at the base point, but the previous iteration's
**accepted** `evalLnL` already built `d_echild`/`d_partial` for exactly that brlen/alpha/pinv and nothing dirties
them. ⏳ **~0.44 s/iter** (one redundant `postorderFill` = 98 `k1_node` launches at 1M), every iteration.
**Fix:** a "device matches base" flag set by an accepted `evalLnL`, reset by any rejected one; skip the rebuild
when set. **Risk:** med (intra-call state tracking; the stateless-across-calls coherence contract is unaffected).

### #3 — Fuse `kj_theta`/`kj_derv`/`kj_ratenum` (CONFIRMED, secondary). `gpu_lnl_intree.cu:494–527`
`edgeThetaInto` materializes the full 601 MB `d_theta`, then `kj_derv` and `kj_ratenum` each re-read it. ⏳ traffic
3 GB/edge → ~1.2 GB/edge if `derv`+`ratenum` read `node`/`dad` directly (as `k2_derv` already does on-the-fly),
saving ⏳ **~0.4 s/sweep**. Keep `kj_theta` for `evalLnL` (no ratenum there). **Risk:** med (new fused kernel).

### #4 — pinv finite-difference does a full extra `evalLnL` (CONFIRMED, +I only). `gpu_lnl_intree.cu:557`
Every +I iteration spends a full sweep (⏳ ~0.45 s) for one scalar dlnL/dpinv. The "reuse base partials" shortcut
is **blocked** because `applyPinv` rescales `catRate=meanR/(1−pinv)`, so pinv enters every transition matrix (not
just the additive invariant term). An analytic dlnL/dpinv is *possible* (the rate-coupling term is most of `gradR`,
already computed) but non-trivial. Lower priority; +I models only.

### #5 — Concurrent-stream multi-edge (PARTIAL, higher effort). `gpu_lnl_intree.cu` syncs at 503–532
Each kernel is occupancy-capped at ~49% warps; independent edges (edge-v reduce vs edge-(v+1) `kj_pre`, and
siblings) could co-reside on separate streams to approach full occupancy — *different* from the K3 CUDA-graph
result (which only collapsed host submission and gave parity). The current blocking `cudaDeviceSynchronize` after
every launch serializes the dependent chain. **Try only after #1.** **Risk:** med–high.

### Refuted / minor (honest negatives)
- **`rebuildEchild` host exp()+H2D — REFUTED as a 1M bottleneck.** Work is `nnodes·ncat·ns²` (≈315K) + a 2.5 MB
  H2D — **nptn-INDEPENDENT**, ~few ms, flat from 100K→1M. Not worth kernelizing for wall.
- **Per-edge `std::vector`/`cudaMemcpyToSymbol` (`setVal` 480, `rs`/`g_rscale` 526–527, `bi(nptn)` 436):**
  O(ncat·ns), nptn-independent (`bi` is once/model) → ms-scale; cleanup, negligible wall.
- **No redundant topology rebuilds** — DFS/`child`/`postorder`/`slot`/`edgeV` built once (404–416). Good.
- `gradR` reduction (534) and `invL` exp loop (529) are the same host-reduction family as #1 → fold in when #1 lands.

---

## VIII.3 Memory (scalability ceiling — see PART VII)

`gbj_partial = nInternal · slotSz` keeps **all 98 postorder partials resident ≈ 59 GB at 1M**; with the preorder
pool (already O(depth), ~26 GB) and scratch, **~85–88 GB/model** — exceeds A100-80, fits only H200-141. The
two-pass Ji gradient inherently needs the postorder partials live during the preorder, so naive recycling does not
apply; **pattern tiling (PART VII) is the fix** — exact, tunable (T=10 → 8.9 GB fits V100/RTX4090), kernel-reuse.
This gates a *clean* sub-H200 1M run independent of every wall lever above.

---

## VIII.4 MEASURED (job 170517590, H200 AA-1M) — and the new #0 lever the static audit could not see

**The measurement landed: 1 H200 CTF = 1994 s = 1.54× faster than 2 SPR nodes (np2 3076.9 s); winner LG+G4
(correct).** Misses 4 nodes (np4 1974.5 s) by 1 %. Refine decomposition exposed a lever **bigger than every
§VIII.2 kernel finding**, invisible to a static kernel read because it lives at the IQ-TREE *call* level:

### #0 — Redundant +I restart sweep (NEW, biggest, algorithmic). `model/modelfactory.cpp:optimizeParametersGammaInvar`
IQ-TREE's "**Thoroughly optimizing +I+G parameters from 10 start values**" (modelfactory.cpp:1453) sweeps 10
initial pinv values and calls `optimizeParameters` (→ JOLT) **once per start**. MEASURED (H200 1M): LG+I+G4 = **869 s
(10 JOLT calls)**, LG+F+I+G4 = **889 s (10 calls)**, vs LG+G4 = **78 s (1 call)** — an **11× gap, ~87 s/call**; on
the 2.2 %-constant 1M data all 10 converged to the identical pinv→1e-6 optimum.

**⚠️ CORRECTION (measured, job 170579044) — full single-start is UNSAFE; the fix is FEWER starts, not ONE.** I
first implemented "single JOLT call, skip all restarts" on the premise that JOLT's joint LM finds the global
optimum from any start. **The validation gate refuted it:** on collapsed data (pinv→0) single-start PASSED (6 s vs
75 s, MLE matched), but on a **high-pinv synthetic (pinv≈0.5)** single-start **converged to pinv=0.457, lnL 39.5
nat BELOW** the 10-start optimum (pinv=0.500). Diagnosis: JOLT's joint optimiser is reliable for **small** pinv
moves but **stalls on large pinv travel** (FD-pinv gradient + joint LM damping declare convergence early) — which
is exactly the pinv ridge the multi-start exists to defeat. **The restart sweep is NOT pure waste; it is the
robustness mechanism, and on collapsed data it merely *looks* redundant.** **Adopted fix (G.4.3c):** under
`--jolt`, reduce the spanning starts **10 → 4** (`n_pinv_starts`), not 1 — one of the 4 always lands near any
optimum and JOLT polishes it locally (where it is reliable). ⏳ **+I refine ~870 s → ~350 s ⇒ CTF total ~936 s =
3.3× vs np2 / 2.1× vs np4 — still beats both bars, and is ROBUST.** Gate (re-validating, job 170580368): 4-start
+I+G MLE == 10-start MLE on collapsed AND high-pinv. **Lesson banked: the multimodal gate is mandatory — it caught
a 39.5-nat correctness regression a collapsed-only test would have shipped.**

### Recommended order (re-ranked by the measurement)
1. **#0 skip redundant +I restarts** — ⏳ ~5× total; algorithmic; the clear path past 4 nodes. Validate global-opt
   robustness first.
2. **#1 on-device reduction** — ⏳ ~1.6× the gradient sweep (GPU util was **62 %**, confirming the host-reduction
   idle); low-med risk. Alone: ⏳ total ~1306 s = 1.51× vs np4.
3. **#2 skip redundant base sweep**, **#3 fuse theta/derv/ratenum** — compound with #1.
4. **PART VII pattern tiling** — orthogonal (memory). NOTE the measured peak was **67.8 GB** for 946,439 patterns
   (< the 88 GB estimate in §VIII.3 / PART VII — re-state to ~68 GB), so **A100-80GB is viable** (run 170575806);
   tiling is still required for V100/consumer.
5. **#4 analytic pinv / #5 concurrent streams** — last.

**Bottom line (measured):** correctness banked; **1 H200 beats 2 nodes (1.54×) today, un-optimized**; the +I
restart redundancy (#0) and the on-device reduction (#1) are two independent levers that each push past 4 nodes.
The 2-node bar is met; 4 nodes is one validated optimization away.

---

## VIII.5 SECOND-ROUND AUDIT (2026-06-13) — the fused kernel reviewed honestly, and where the wall actually is

After G.5.0 Part A+B and the gradient-kernel fusion landed, the user asked the right question: *the fusion barely
moved the wall — is `kj_derv_fused` optimal, or can it be improved?* The honest answer needed hard data, so I read the
**per-kernel register/occupancy out of the frozen cubin** (`cuobjdump --dump-resource-usage`, sm_80/A100):

| kernel | role | **REG (sm_80)** | spill (STACK) | occupancy implication (256-thr block, A100 64K reg/SM) |
|---|---|---:|---:|---|
| **`k1_node`** | postorder lnL (the DOMINANT sweep) | **56** | **160 B** | 65536/(256·56)=**4 blocks/SM = 32/64 warps = 50 %** ← the binding bound |
| `kj_pre` | preorder gradient partial | 48–56 | 320 B | ~50 % (same ceiling) |
| **`kj_derv_fused`** | fused theta+derv+ratenum | **32** | 0 | 65536/(256·32)=**8 blocks/SM = 64/64 warps = 100 %** |
| `kj_derv` / `kj_theta` / `kj_ratenum` (now dead) | the 3 it replaced | 32 each | 0 | 100 % each |
| `k2_derv`, `k_leaf_eig`, reductions | scaffolding | 31–40 | 0 | ≥80 % |

**Verdict on `kj_derv_fused`: it is essentially optimal for what it is, and the small wall effect is EXPECTED, not a
defect.**
1. **Register-neutral (32 = the same as the 3 separate kernels).** My pre-measurement worry — that fusing would raise
   register pressure and *lower* occupancy (the real bound) — is **refuted by the cubin**: the compiler keeps the fused
   inner loop at 32 regs (theta lives in one register `th`, consumed immediately into the lh/rc/d2 accumulators). So
   the fusion bought the bandwidth saving **for free**, no occupancy penalty. At 32 regs it can reach 100 % occupancy.
2. **Memory access is already ideal:** `node[o]`/`dad[o]` with `o=(c·ns+x)·nptn+ptn` are **coalesced** (adjacent
   threads → adjacent `ptn` → adjacent addresses); `g_val0/1/2`/`g_rscale` are **constant-cache broadcasts** (loop-
   uniform); `node`+`dad` are each read **exactly once** per element (the theoretical minimum). There is no further
   fusion or access-pattern change to *this* kernel that would matter.
3. **Why it barely moved the wall — the K4 lesson, re-confirmed by measurement.** The fusion attacks **VRAM traffic**
   (it removed the 601 MB `d_theta` round-trip = 5→2 transactions/element ≈ 2.5×) and **launch count** (3→1 launches/
   edge). But **P3.0b measured the binding resource as memory *latency* + scheduler starvation on `k1_node`** (DRAM
   33 %, issue_active 7.8 %, 50 % occ), and **K3 showed launch latency is already hidden**. Fusion saved two
   non-binding resources. It is a **correct, clean keeper** (real traffic cut + 601 MB VRAM reclaimed + tidier code),
   **not a wall lever** — exactly as the K4 fusion (98→42 launches) was wash-to-loss at 100K for the same reason.

**Where the wall actually is — `k1_node`'s 56-register / `STACK:160` spill.** The dominant postorder kernel carries a
per-thread `double prod[NS_MAX]` (`NS_MAX=20` ⇒ 160 B), which **spills to local memory and pushes it to 56 regs ⇒
4 blocks/SM ⇒ 50 % occupancy** — *precisely* the P3.0b bound (the kernel is latency-bound and the schedulers sit idle
~92 % of cycles for want of eligible warps). **The only lever on the real bound is the P2∥ occupancy attack:**
thread-per-(pattern×category×output-state), which collapses `prod[20]` to a scalar accumulator, eliminates the spill,
and drops registers below the 32-reg / 8-block / 100 %-occupancy line — potentially **doubling** the resident warps
that hide the memory latency. It is HIGH-effort and an **honest coin-flip** (the NS=20 matvec may be too small to
amortize the cross-state reduction), but it is the one change that touches what P3.0b proved is binding. **More
gradient-side fusion cannot help; occupancy can.**

**Two easy wins banked this round (validating, job 170730082):**
- **part8 #2 base-sweep skip** — `computeGradient` no longer rebuilds echild + re-runs the postorder at the base point
  when the immediately-preceding accepted `evalLnL` already built exactly that point (guarded by an exact
  (brlen,α,pinv) comparison; falls back to rebuild on any mismatch). This skips **~98 `k1_node` launches per gradient
  iteration** — i.e. one full sweep of *the bottleneck kernel* every iter — and is **bit-exact** (it removes a
  redundant identical computation). This is a *real* wall win, unlike the fusion, because it removes `k1_node` work.
- **`d_theta` reclaim** — the 601 MB theta arena is dead post-fusion; no longer allocated (VRAM/capability margin for
  +R10 / A100-80, not wall).
