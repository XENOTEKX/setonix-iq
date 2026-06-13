#!/usr/bin/env python3
"""analyze_subsuff.py — PART X §X.3.2 test: does CTF's (N/m) scale-consistent BIC PROJECTION hurt top-k recall of a
low-parameter true winner vs. ranking by the NATIVE subsample BIC? Reads the .iqtree tables produced by
run_subsample_sufficiency_sweep.sh and reports recall(top-3 contains FULLWIN) under BOTH rankings, per length, with
the ΔBIC margins and the recovered parameter counts. Re-runnable on the saved files (no recompute).

  python3 analyze_subsuff.py [SWEEP_DIR=/scratch/rc29/as1708/iqtree3-gpu/subsuff_sweep] [FULLWIN=LG+G4] [NFULL=1000000]

Why this exists: the native subsample BIC penalises params at ln(L); the projection re-credits per-site lnL by N/L,
which AMPLIFIES the n-independent overfit excess of high-parameter (+R) models ~ (N/L)x while leaving the penalty
p*ln(N) unchanged. Theory (§X.3.2) predicts the projection ranks low-p winners WORSE. This measures it.
"""
import sys, re, glob, os, math

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "/scratch/rc29/as1708/iqtree3-gpu/subsuff_sweep"
FULLWIN = sys.argv[2] if len(sys.argv) > 2 else "LG+G4"
NFULL = int(sys.argv[3]) if len(sys.argv) > 3 else 1000000   # full alignment length (sites); ranking-neutral to sites-vs-patterns

# .iqtree "List of models sorted by BIC scores": Model LogL AIC <+/-> w-AIC AICc <+/-> w-AICc BIC <+/-> w-BIC
# BUGFIX 2026-06-13: the table has an explicit +/- sign token between each value and its weight; the prior regex
# (\s+\S+ for the weight) failed to match EVERY row -> empty tables (the SAME bug was in the sweep job's embedded
# analysis, so the sweep's printed tables were empty — Sonnet's §X.5 numbers were derived elsewhere). groups:
# 1=model 2=logL 3=AIC 4=AICc 5=BIC.
ROW = re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')

def parse(iq, L):
    """Return list of (name, logL, bic_native, p) recovering p from the native subsample BIC at n=L."""
    out = []
    lnL_n = math.log(L)
    for line in open(iq):
        m = ROW.match(line)
        if not m:
            continue
        name, logl, bic = m.group(1), float(m.group(2)), float(m.group(5))
        p = round((bic + 2 * logl) / lnL_n)   # bic = -2logl + p*ln(L)  =>  p
        out.append((name, logl, bic, p))
    return out

def rank_recall(rows, key):
    rows2 = sorted(rows, key=key)
    names = [r[0] for r in rows2]
    rank = names.index(FULLWIN) + 1 if FULLWIN in names else -1
    top3 = FULLWIN in names[:3]
    margin = (key(rows2[1]) - key(rows2[0])) if len(rows2) > 1 else float('nan')
    return names[0], rank, top3, margin

def run(glob_pat, label):
    files = glob.glob(glob_pat)
    if not files:
        print(f"\n── {label}: no files ({glob_pat}) ──")
        return
    print(f"\n── {label} ──")
    print(f"{'L':>7} {'seed':>4} | {'NATIVE winner':16} {'rk':>3} {'top3':>5} {'ΔBIC':>10} | "
          f"{'PROJECTED winner':16} {'rk':>3} {'top3':>5} {'ΔBIC':>10} | {'p('+FULLWIN+')':>8}")
    by = {}
    def lk(f):
        mm = re.search(r'_(\d+)_(\d+)\.iqtree', os.path.basename(f)); return (int(mm.group(1)), int(mm.group(2)))
    for f in sorted(files, key=lk):
        L, seed = lk(f)
        rows = parse(f, L)
        if not rows:
            continue
        nw, nr, nt, nm = rank_recall(rows, key=lambda r: r[2])                                  # native BIC
        pw, pr, pt, pm = rank_recall(rows, key=lambda r: -2*(NFULL/L)*r[1] + r[3]*math.log(NFULL))  # projected
        pwin = next((r[3] for r in rows if r[0] == FULLWIN), None)
        print(f"{L:>7} {seed:>4} | {nw:16} {nr:>3} {str(nt):>5} {nm:10.1f} | "
              f"{pw:16} {pr:>3} {str(pt):>5} {pm:10.1f} | {str(pwin):>8}")
        by.setdefault(L, []).append((nt, pt, nw == FULLWIN, pw == FULLWIN))
    print(f"\n  {'L':>7} | {'native recall':>13} {'native exact':>12} | {'proj recall':>11} {'proj exact':>10}")
    for L in sorted(by):
        v = by[L]; n = len(v)
        print(f"  {L:>7} | {sum(e[0] for e in v)/n:13.2f} {sum(e[2] for e in v)/n:12.2f} | "
              f"{sum(e[1] for e in v)/n:11.2f} {sum(e[3] for e in v)/n:10.2f}")

print(f"PART X §X.3.2 — native vs projected-BIC recall of {FULLWIN}  (NFULL={NFULL}, dir={SWEEP})")
run(f"{SWEEP}/test_*_*.iqtree", "-m TEST (no +R)")
run(f"{SWEEP}/mf_*_*.iqtree",   "-m MF (+R present — the projection stress test)")
print("\nVERDICT KEY:")
print("  CTF is licensed at the smallest L where the chosen ranking's recall=1.00 across resamples.")
print("  If projected recall < native recall (esp. in -m MF where +R appears), §X.3.2 is CONFIRMED:")
print("  the (N/L) projection over-credits high-p models and the native subsample BIC is the safer top-k gate.")
