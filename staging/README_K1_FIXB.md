# K1 / Fix-B — preserved, NOT merged into the default path (2026-06-27 consolidation)

These two files are a **deliberately preserved alternate version** of the GPU likelihood/optimizer
translation unit `tree/gpu/gpu_lnl_intree.cu`, kept here so no validated work is lost during the
GPU-branch consolidation. They are NOT compiled into the unified binary.

- `gpu_lnl_intree.cu.prefixb` — the gradient-loop baseline (before Fix-B / K1).
- `gpu_lnl_intree.cu.fixb`    — baseline **+ K1 "Fix-B" val-table hoist** (byte-identical to the
  `tree-search-ts0` working tree in the sibling clone `/scratch/rc29/as1708/iqtree3-gpu`).

## What K1 / Fix-B is
The TS.2.1 "val-table hoist": replaces the per-edge `cudaMemcpyToSymbol(g_val0/1/2/g_rscale)` +
redundant `cudaDeviceSynchronize` with a single bulk H2D of every edge's `{v0,v1,v2,rscale}` tables
(`gbj_val0all/val1all/val2all/rscaleall`) consumed by a new kernel `kj_derv_fused_arr` (identical
inner-loop arithmetic to `kj_derv_fused`, only the coefficient *source* differs: `__constant__` →
`__global__` arg). Host-scheduling change only — **zero arithmetic difference**.

## Why it is preserved but NOT on the default path
- **Perf-FLAT / thesis FALSIFIED** (tree-search memory, job 172267130): screener_wall 128.318 vs
  128.205 s = **1.001× (noise)**; the val-table memcpy was never on the screener critical path. K1 is
  a safe *cleaner*, not a speed win.
- **The unified default gradient loop carries the validated +R (FreeRate) path** (G.5.1b, job
  172444201, 3/3 PASS). The +R weight-gradient capture (`kj_reduce_gradnum` at the first edge,
  `accW`/`accWk` Kahan) lives *inside* the per-edge loop that Fix-B restructures. Re-applying Fix-B
  into the +R loop would put an **unvalidated +R×FixB interaction** on the science path for a
  perf-neutral gain — not worth the risk to a validated path.

## If K1 is ever wanted on the default path
Re-apply the Fix-B batched-coeff sweep *into* the current +R-bearing loop in
`tree/gpu/gpu_lnl_intree.cu` (do NOT `cp` this file over — it lacks +R and L-BFGS), gate it behind an
env flag `JOLT_K1_HOIST` defaulting OFF, and validate `K1-ON == K1-OFF` bit-identical **on a +R
case** before defaulting on. Given the falsified perf thesis, this is low priority.
