#!/usr/bin/env python3
"""ctf_rerank.py — the CTF coarse-stage gate, in ONE tested place.

Single source of truth for the native-subsample-BIC rerank + rate-heterogeneity detector + per-model
refine/skip action that every CTF benchmark script (run_ctf_1m_mf_energy.sh, run_ctf_1m_mf_dna_energy.sh,
run_ctf_10m_mf_aa_h200.sh, run_ctf_freerate_recall_sweep.sh, ...) currently re-implements inline as a
heredoc. Lifting it here (part9 IX.7.1 #4 / IX.10.1 #5: "port the native-BIC gate + detector + budget into
the production CTF path, not just the benchmark script") deduplicates the §X.5.5 fix and lets it be
regression-tested with `--selftest` rather than re-derived per script.

WHY native BIC, not the projection (the §X.5.5 / jobs 170728179,182 bug):
  The OLD scale-consistent PROJECTION  -2*(N/m)*logL + p*ln(N)  amplifies any subsample logL diff by 2*N/m
  (~378x at m=5000, N=1e6) while leaving the p*ln(N) penalty fixed => it over-credits the +I/+R subsample
  overfit and ranked [LG+I+G4, LG+R5, LG+I+R5] ABOVE the true winner LG+G4 (recall FAIL -> +R refined on
  CPU at 1M -> walltime). On a 5000-site subsample every candidate is within <1 nat of FIT, so the
  rate-model choice is decided ENTIRELY by the penalty => the NATIVE subsample BIC (penalty ln m, the BIC
  column IQ-TREE already prints) is the right gate and ranks LG+G4 #1 (verified across the PART X sweeps).
  We rank ALL candidates (no pre-exclusion — don't hide the +R coverage gap), add a rate-het DETECTOR, and
  emit a refine/skip action so an ineligible +R CPU-refine at 1M cannot blow the wall (the caller caps each
  refine with `timeout`).

CLI (drop-in for the old heredoc):
    python3 ctf_rerank.py <coarse.iqtree> <m_subsample> <N_full> <TOPK>
  -> stdout : one line per top-k model   MODEL:<name>:<refine|skip>
  -> stderr : [rerank] OLD projected top-5 / NATIVE BIC top-5 ; [detector] ... RATE_HET_FLAG=<bool>
Self-test:
    python3 ctf_rerank.py --selftest      # pins the §X.5.5 ranking flip + both detector branches
"""
import sys, re, math

# The IQ-TREE "List of models sorted by BIC" table row: Model LogL AIC +w AICc +w BIC +w  (groups: name,
# logL, AIC, AICc, BIC). Byte-identical to the regex the bench heredocs used — do not "tidy" it.
ROW = re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')


def ineligible(name):
    """Mirrors the in-tree JOLT eligibility gate (part9 IX.1): FreeRate (+R / +I+R) and pure-+I decline to
    CPU; everything else (incl. +I+G, free-Q DNA) engages JOLT. Keep in lockstep with phylotreegpu.cpp."""
    return ('+R' in name) or ('+I' in name and '+G' not in name)


def parse(iqtree_path):
    """Parse the coarse .iqtree model table -> list of rows. p (param count) is recovered exactly from the
    native BIC identity  bic = -2*logL + p*ln(m)  =>  p = (bic + 2*logL)/ln(m)  (no separate p column to
    parse, and it round-trips bit-exactly for the integer p)."""
    rows = []
    for line in open(iqtree_path):
        mm = ROW.match(line)
        if not mm:
            continue
        name = mm.group(1)
        logl = float(mm.group(2))
        bic = float(mm.group(5))   # group(5) = the BIC column = NATIVE subsample BIC (penalty ln m)
        rows.append((name, logl, bic))
    return rows


def rerank(rows, m, N, K):
    """Return (nat_topk, diag) where nat_topk is [(name, action), ...] for the top-K by native BIC and diag
    carries the projected top-5 (diagnostic only) + the detector result. Pure function — no I/O."""
    enr = []
    for name, logl, bic in rows:
        p = round((bic + 2.0 * logl) / math.log(m))                 # native-BIC param-count recovery
        proj = -2.0 * (N / m) * logl + p * math.log(N)              # the OLD projection (kept for diag only)
        enr.append({'name': name, 'logl': logl, 'bic': bic, 'p': p, 'proj': proj, 'inel': ineligible(name)})

    nat_all = sorted(enr, key=lambda r: r['bic'])                   # NATIVE BIC over ALL candidates = the gate
    proj_top5 = [r['name'] for r in sorted(enr, key=lambda r: r['proj'])[:5]]

    be = min((r for r in enr if not r['inel']), key=lambda r: r['bic'], default=None)  # best eligible
    bi = min((r for r in enr if r['inel']),     key=lambda r: r['bic'], default=None)  # best ineligible (+R/+I)
    detector = None
    if be and bi:
        margin = abs(bi['p'] - be['p']) / 2.0       # ~Δp/2 nats AIC overfit cushion
        lead = be['bic'] - bi['bic']                # >0 => an ineligible (+R/+I) model LEADS the eligible best
        detector = {'be': be, 'bi': bi, 'margin': margin, 'lead': lead, 'flag': lead > margin}

    out = []
    for r in nat_all[:K]:
        # refine eligible models always; refine an INELIGIBLE (+R/+I) model only if it could plausibly win
        # (within the overfit margin of the best eligible) — else SKIP it (detector-justified): it cannot be
        # the full-data winner, so we don't burn the CPU-at-1M budget on a doomed refine.
        skip = r['inel'] and (be is not None) and (r['bic'] > be['bic'] + abs(r['p'] - be['p']) / 2.0)
        out.append((r['name'], 'skip' if skip else 'refine'))
    return out, {'nat_top5': [r['name'] for r in nat_all[:5]], 'proj_top5': proj_top5, 'detector': detector}


def main_cli(argv):
    iq, m, N, K = argv[1], int(argv[2]), int(argv[3]), int(argv[4])
    rows = parse(iq)
    topk, diag = rerank(rows, m, N, K)
    sys.stderr.write("  [rerank] OLD projected top-5 (the bug): " + ", ".join(diag['proj_top5']) + "\n")
    sys.stderr.write("  [rerank] NATIVE BIC top-5 (the gate):   " + ", ".join(diag['nat_top5']) + "\n")
    d = diag['detector']
    if d:
        sys.stderr.write(f"  [detector] best_elig={d['be']['name']}({d['be']['bic']:.1f}) "
                         f"best_inel={d['bi']['name']}({d['bi']['bic']:.1f}) inel_lead={d['lead']:.1f} "
                         f"margin={d['margin']:.1f} RATE_HET_FLAG={d['flag']}\n")
        if d['flag']:
            sys.stderr.write("  [detector] *** WARNING: a +R/+I model genuinely leads on the subsample — "
                             "eligible-refine may MISS the true winner; needs G.5.1 (+R JOLT) or CPU "
                             "full-refine ***\n")
    for name, action in topk:
        print(f"MODEL:{name}:{action}")
    return 0


# ----------------------------------------------------------------------------------------------------------
def _fixture(lines):
    """Build rows from synthetic .iqtree table lines (selftest). Each line is the exact format ROW expects:
    NAME LOGL AIC + wAIC AICc + wAICc BIC + wBIC."""
    import io
    rows = []
    for ln in lines:
        mm = ROW.match(ln)
        assert mm, f"fixture line did not match ROW: {ln!r}"
        rows.append((mm.group(1), float(mm.group(2)), float(mm.group(5))))
    return rows


def _bic(logl, p, m):
    return -2.0 * logl + p * math.log(m)


def selftest():
    m, N = 5000, 1_000_000
    lm = math.log(m)
    ok = True

    def line(name, logl, p):
        b = _bic(logl, p, m)
        # AIC/AICc columns are irrelevant to the gate; fill plausibly so ROW matches.
        a = -2.0 * logl + 2.0 * p
        return f"{name} {logl:.3f} {a:.3f} + 0.0 {a:.3f} + 0.0 {b:.3f} + 0.0"

    # --- Fixture A: the §X.5.5 bug — native ranks LG+G4 #1; projection promotes +I/+R above it -----------
    A = _fixture([
        line("LG+G4",     -300000.000, 198),
        line("LG+I+G4",   -299999.500, 199),
        line("LG+R5",     -299999.000, 206),
        line("LG+I+R5",   -299998.800, 207),
        line("WAG+G4",    -300500.000, 198),
    ])
    topk, diag = rerank(A, m, N, K=4)
    assert diag['nat_top5'][0] == "LG+G4", diag['nat_top5']
    # projection must put LG+G4 OUT of the top-3 (the recall miss the native gate fixes)
    assert "LG+G4" not in diag['proj_top5'][:3], diag['proj_top5']
    assert diag['proj_top5'][0] in ("LG+I+R5", "LG+R5"), diag['proj_top5']
    # detector must NOT fire (LG+G4 genuinely wins); the +R model in top-k must be SKIPPED
    assert diag['detector']['flag'] is False, diag['detector']
    acts = dict(topk)
    assert acts["LG+G4"] == "refine" and acts["LG+I+G4"] == "refine", acts
    assert acts["LG+R5"] == "skip", acts                 # ineligible, behind best eligible by > margin
    print(f"  [selftest A §X.5.5] native#1={diag['nat_top5'][0]}  projected_top3={diag['proj_top5'][:3]}  "
          f"detector.flag={diag['detector']['flag']}  LG+R5={acts['LG+R5']}  -> PASS")

    # --- Fixture B: genuinely rate-heterogeneous — a +R model leads by > margin => detector FIRES ---------
    B = _fixture([
        line("LG+G4", -300100.000, 198),     # eligible
        line("LG+R4", -300000.000, 204),     # ineligible, fits 100 nat better, only +6 params
    ])
    topk, diag = rerank(B, m, N, K=2)
    assert diag['nat_top5'][0] == "LG+R4", diag['nat_top5']        # +R wins native BIC here
    d = diag['detector']
    assert d['flag'] is True, d                                   # lead (~148) >> margin (3) => FIRES
    acts = dict(topk)
    assert acts["LG+R4"] == "refine", acts                        # the leader is refined, never skipped
    print(f"  [selftest B rate-het] native#1={diag['nat_top5'][0]}  lead={d['lead']:.1f} margin={d['margin']:.1f} "
          f"RATE_HET_FLAG={d['flag']}  LG+R4={acts['LG+R4']}  -> PASS")

    # --- Fixture C: param-count recovery is exact for the integer p (no off-by-one from float ln) ---------
    for p_true in (1, 99, 198, 206, 999):
        b = _bic(-123456.789, p_true, m)
        p_rec = round((b + 2.0 * (-123456.789)) / lm)
        assert p_rec == p_true, (p_true, p_rec)
    print("  [selftest C p-recovery] exact for p in {1,99,198,206,999}  -> PASS")

    # --- Fixture D: DNA — F81+F+G4 wins native BIC (the G.6.2 1M result), GTR demoted by penalty ----------
    D = _fixture([
        line("F81+F+G4",    -298000.000, 102),
        line("HKY+F+G4",    -297999.700, 103),
        line("GTR+F+G4",    -297999.500, 107),    # 0.5 nat better fit than F81, +5 free-Q params
        line("GTR+F+I+G4",  -297999.400, 108),
    ])
    topk, diag = rerank(D, m, N, K=3)
    assert diag['nat_top5'][0] == "F81+F+G4", diag['nat_top5']     # penalty demotes GTR (G.6.2 vindication)
    assert all(a == "refine" for _, a in topk), topk              # all free-Q DNA models are JOLT-eligible
    print(f"  [selftest D DNA free-Q] native#1={diag['nat_top5'][0]}  top3={diag['nat_top5'][:3]}  -> PASS")

    print("ALL SELFTESTS PASS" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if len(sys.argv) != 5:
        sys.stderr.write(__doc__)
        sys.exit(2)
    sys.exit(main_cli(sys.argv))
