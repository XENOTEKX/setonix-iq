# Track A — tip-vector precompute (leaf-fold `tabP`), Phase-0 gate results

**2026-07-18. Compile-only + echild-tax steps done; bit-identity + wall-time steps NOT yet run.** AA-side depth lever
(the standing plan `generic-brewing-feather.md` Track A). Assistant does NOT push GPU source.

## Step 1 — echild-rebuild tax (ovlevers 174087249, the plan's "first cheap action")
The host per-eval echild rebuild is **NEGLIGIBLE**: DNA-100K mean **0.003s/call** (Σ0.009s / 3 calls, 210 rebuilds),
AA-100K mean **0.006s/call** (Σ0.012s / 2 calls, run timed out early). ⇒ the plan's "one unmeasured risk" — that building
`tabP` host-side would compound an existing echild tax — is **REFUTED**; the tax is milliseconds. **`make_tabP`
host-vs-device placement is de-risked** (both fine on the tax axis; still build it as a device kernel per the plan for
Amdahl headroom as the GPU-side win grows).

## Step 3 — compile-only register/spill gate (login node, no GPU; `tabp_gate.cu`)
Byte-faithful copy of the shipped `accum_child_t` (`gpu_lnl_intree.cu:608-627`) + `k1_node_t` (`:630-659`) as BASELINE,
plus the `tabP`-gather leaf variant (leaf `p==null` fold → `prod[x]*=tabP[(c*NS+x)*(NS+1)+s]`, ambiguous bucket `s=NS`;
internal `p!=null` matvec UNCHANGED). `nvcc -Xptxas -v -arch=sm_90`:

| kernel | registers | spill |
|---|---|---|
| `k1_node_t<4>` baseline (DNA) | 48 | 0 |
| `k1_node_t<20>` baseline (AA) | **128** | 0 |
| `k1_node_tabP<4>` (DNA) | 48 | 0 |
| `k1_node_tabP<20>` (AA) | **128** | **0** |

`-maxrregcount` sweep (does tabP FREE registers ⇒ occupancy upside?): **cap 96 → baseline and tabP IDENTICAL** (8B
stack / 24 spill-store / 16 spill-load each); **cap 64 → tabP marginally less** (264 vs 272B stack; a wash).

**✅ GATE PASSES — but with a tempered upside (honest):**
- **No regression:** `k1_node_tabP<20>` = 128 reg / 0 spill, EXACTLY the shipped `k1_node_t<20>`. The lever is NOT killed
  by register pressure (the plan's kill criterion).
- **No occupancy upside:** tabP is register-EQUIVALENT, not cheaper. The 128-reg ceiling is set by the *internal*-child
  O(ns²) matvec + the non-root output matvec — which tabP leaves untouched — so removing the leaf matvec frees nothing at
  the ceiling. ⇒ the win (if any) is **pure instruction-count on leaf patterns**, NOT higher occupancy.
- **Consequence for the projection:** the plan's "~1.1–1.4×, memory-latency-bound" is now bounded on TWO sides — the
  benefit only reaches the ≈50% of edges that are pendant (leaf), and only converts to wall if the kernel is
  INSTRUCTION-bound on the leaf fold (if it's memory-latency-bound on the partials load, dropping FMAs does little). The
  ncu profile (step 5) settles which regime it's in.

## 🔴 THE PLAN'S "Phase 0" IS LARGELY ALREADY DONE — corrected 2026-07-18 (THE MAINSTAY caught it before a redundant GPU job)
The standing plan (`generic-brewing-feather.md`) framed Track A as greenfield. It is NOT. A prior session (2026-07-10,
memory `project_tipvec_stallsplit`) already implemented, profiled, and MEASURED this exact lever:
- **Implemented + committed:** `4de1e6c4` ("LEVER 3 tip-vector precompute, merge onto gpu-kernel-dev, default-OFF"), a
  compile-time `template<int NS, bool TIPVEC>` — **already merged into the CURRENT trees** (`iqtree3-mfresident`,
  `iqtree3-l2search-stage2b`, `iqtree3-mfdevcheck`), SASS-identical to the shipped kernel when OFF (all 4096 sm_90
  insts). Worktree `iqtree3-tipvec` binary `dcc970d0`. Assistant did NOT push. `tabP`-invalidation-on-mixture bug and an
  atexit-counter false-negative bug both found+fixed there.
- **ncu bimodal stall split ALREADY MEASURED (job 173279955, `k1_raw.csv`):** pop A (internal-child folds, `pc[i*nptn]`
  strided global load → **long_scoreboard 7.14cyc / memory-latency-bound**, L1 64%) = **53.8%** of kernel time,
  UNTOUCHABLE by tip-vec; pop B (leaf-child folds, register-indexed `__constant__` `g_Uinv[i*NS+s]` → constant-cache
  serialization → **short_scoreboard 3.62cyc**, L1 85%) = **46.2%**, the tip-vec target. SASS OFF→ON: `LDC 49→16`,
  `LDG 1922→1323`. So tip-vec is a PURE pop-B lever ⇒ Amdahl ceiling 1.10–1.19×, **measured end-to-end 1.075×** (job
  173071576). Registers `<20,true>`=126 vs `<20,false>`=128 ⇒ **no occupancy gain** — which THIS gate independently
  re-derived (128/0, register-neutral). My compile gate corroborates prior work; it did not discover anything new.
- **Structural ceiling (durable):** tip-vec exists ONLY because a leaf has `ns+1` discrete states → a finite table. An
  internal child's partial is a continuous `nptn×ns` vector → no table → the memory-bound pop A (54%) can NEVER be
  addressed this way. tip-vec is intrinsically capped near ~1.1×.

**Bottom line (corrected):** tip-vec is a **DONE, characterized, modest 1.075× AA lever**, already sitting as default-OFF
`4de1e6c4` in the current trees. There is NO Phase-0 spike to run — the `ncu`/bit-identity steps are already answered. The
only open action is a **graduate-or-leave decision**: to ship it, re-validate `4de1e6c4` default-ON on the CURRENT
promoted binary (bit-identity + one AA-1M wall to confirm 1.075× holds post-NS-template-merge), then graduate. Given the
1.075× size and DNA being the priority, this is low-urgency. My compile gate + echild-tax steps stand as corroboration,
not new Phase-0 findings.
