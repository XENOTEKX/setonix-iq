#!/usr/bin/env python3
# compare_mode_p_parity.py — parity checker for Mode P ISO gates.
#
# Compares two IQ-TREE runs (typically: FCA baseline vs Mode P run) on:
#   - log-likelihood (BEST SCORE FOUND or Log-likelihood of the tree)
#   - Best-fit model (from BIC selection)
#   - BIC value (from the .iqtree report table for the best model)
#   - Wall-clock time for ModelFinder
#   - [Mode P] partition lines (presence + per-rank ranges)
#
# Usage:
#   compare_mode_p_parity.py --ref <baseline-dir-or-file> --test <p3-dir-or-file>
#                            [--tol 1e-6] [--gate iso2]
#
# Exit codes:
#   0  parity PASS for all checks
#   1  parity FAIL for at least one check
#   2  parse error (missing fields in one or both inputs)
#
# Notes:
#   - Accepts either a single iqtree log file or a directory containing
#     iqtree_inner.log + iqtree_stdout.log + iqtree_inner.iqtree
#   - IQ-TREE uses "BEST SCORE FOUND :" with a SPACE before the colon
#   - Mode P log lines have format: "[Mode P] rank R model=X ptn=[a, b) of N"

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple


# ── parsers ─────────────────────────────────────────────────────────────

RE_BEST_SCORE   = re.compile(r"BEST SCORE FOUND :\s*(-?[\d.]+)")
RE_LOG_LH       = re.compile(r"Log-likelihood of the tree:\s*(-?[\d.]+)")
RE_BEST_MODEL_A = re.compile(r"Best-fit model according to (?:BIC|AIC|AICc):\s*(\S+)")
RE_BEST_MODEL_B = re.compile(r"Best-fit model:\s*(\S+)")
RE_MF_WALL      = re.compile(r"Wall-clock time for ModelFinder:\s*([\d.]+)")
RE_MODE_P       = re.compile(r"\[Mode P\] rank (\d+) model=(\S+) ptn=\[(\d+), (\d+)\) of (\d+)")
RE_BIC_TABLE    = re.compile(r"^\s*BIC\s*$", re.MULTILINE)


def collect_files(target: Path) -> List[Path]:
    """Return the list of files to parse for one side of the comparison.
    If target is a file, just that. If a directory, look for the canonical
    set of iqtree output files."""
    if target.is_file():
        return [target]
    if target.is_dir():
        candidates = [
            target / "iqtree_inner.log",
            target / "iqtree_stdout.log",
            target / "iqtree_inner.iqtree",
        ]
        # Also include rank logs if present
        rank_log_root = target / "rank_logs"
        if rank_log_root.is_dir():
            for sub in sorted(rank_log_root.glob("*/stderr")):
                candidates.append(sub)
            for sub in sorted(rank_log_root.glob("*/stdout")):
                candidates.append(sub)
        return [f for f in candidates if f.exists()]
    return []


def read_all(files: List[Path]) -> str:
    """Concatenate the content of all parse-target files into one string."""
    buf = []
    for f in files:
        try:
            buf.append(f.read_text(errors="replace"))
        except OSError:
            pass
    return "\n".join(buf)


def parse_lnl(text: str) -> Optional[float]:
    m = RE_BEST_SCORE.search(text)
    if m:
        return float(m.group(1))
    # fallback to per-tree log-likelihood
    matches = RE_LOG_LH.findall(text)
    return float(matches[-1]) if matches else None


def parse_best_model(text: str) -> Optional[str]:
    m = RE_BEST_MODEL_A.search(text)
    if m:
        return m.group(1)
    m = RE_BEST_MODEL_B.search(text)
    return m.group(1) if m else None


def parse_mf_wall(text: str) -> Optional[float]:
    m = RE_MF_WALL.search(text)
    return float(m.group(1)) if m else None


def parse_mode_p_lines(text: str) -> List[Tuple[int, str, int, int, int]]:
    """Return list of (rank, model_name, ptn_start, ptn_end, ptn_total)."""
    return [
        (int(r), m, int(a), int(b), int(n))
        for r, m, a, b, n in RE_MODE_P.findall(text)
    ]


def parse_bic_from_iqtree_report(report_path: Path, best_model: str) -> Optional[float]:
    """Find the BIC value for best_model in the .iqtree report table.

    The .iqtree report has a table block:
        No. Model         -LnL         df  AIC          AICc         BIC
          5 LG+F          83152896.920 216 166306225.840 166306225.933 166308777.990
    We locate the row whose 2nd column equals best_model and return the last column.
    """
    if not report_path.exists():
        return None
    try:
        lines = report_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        toks = line.split()
        # Need at least 6 tokens: No, Model, -LnL, df, AIC, AICc, BIC
        if len(toks) >= 7 and toks[1] == best_model:
            try:
                return float(toks[-1])
            except ValueError:
                continue
    return None


# ── runner ──────────────────────────────────────────────────────────────

def check_lnl(ref: Optional[float], test: Optional[float], tol: float) -> Tuple[bool, str]:
    if ref is None or test is None:
        return False, f"lnL missing (ref={ref}, test={test})"
    delta = abs(ref - test)
    return (delta <= tol), f"lnL ref={ref}  test={test}  |Δ|={delta:.3e}  tol={tol:.0e}"


def check_model(ref: Optional[str], test: Optional[str]) -> Tuple[bool, str]:
    if ref is None or test is None:
        return False, f"best_model missing (ref={ref}, test={test})"
    return (ref == test), f"best_model ref={ref}  test={test}"


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="Mode P ISO parity checker")
    ap.add_argument("--ref", required=True, type=Path,
                    help="Reference run (FCA baseline or ISO-1 directory/file)")
    ap.add_argument("--test", required=True, type=Path,
                    help="Test run (ISO-0/1/2 directory/file)")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Absolute lnL tolerance (default 1e-6)")
    ap.add_argument("--gate", choices=["iso0", "iso1", "iso2"], default="iso2",
                    help="Which ISO gate semantics to apply")
    ap.add_argument("--report-path-ref",  type=Path,
                    help="Optional path to .iqtree report for ref (for BIC parse)")
    ap.add_argument("--report-path-test", type=Path,
                    help="Optional path to .iqtree report for test")
    args = ap.parse_args(argv)

    ref_files  = collect_files(args.ref)
    test_files = collect_files(args.test)
    if not ref_files:
        print(f"ERROR: ref path '{args.ref}' has no parseable files", file=sys.stderr)
        return 2
    if not test_files:
        print(f"ERROR: test path '{args.test}' has no parseable files", file=sys.stderr)
        return 2

    ref_text  = read_all(ref_files)
    test_text = read_all(test_files)

    ref_lnl   = parse_lnl(ref_text)
    test_lnl  = parse_lnl(test_text)
    ref_best  = parse_best_model(ref_text)
    test_best = parse_best_model(test_text)
    ref_wall  = parse_mf_wall(ref_text)
    test_wall = parse_mf_wall(test_text)
    ref_modep  = parse_mode_p_lines(ref_text)
    test_modep = parse_mode_p_lines(test_text)

    print("══ Mode P parity report ═══════════════════════════════════════")
    print(f"  ref:  {args.ref}  ({len(ref_files)} file(s))")
    print(f"  test: {args.test}  ({len(test_files)} file(s))")
    print(f"  gate: {args.gate}    tol(lnL): {args.tol:.0e}")
    print()

    passes = []

    ok, msg = check_lnl(ref_lnl, test_lnl, args.tol)
    print(("  ✓ " if ok else "  ✗ ") + msg)
    passes.append(("lnL", ok))

    ok, msg = check_model(ref_best, test_best)
    print(("  ✓ " if ok else "  ✗ ") + msg)
    passes.append(("best_model", ok))

    if ref_wall is not None and test_wall is not None:
        delta = test_wall - ref_wall
        pct = 100 * delta / ref_wall if ref_wall > 0 else 0
        print(f"  ℹ MF wall ref={ref_wall:.1f}s  test={test_wall:.1f}s  Δ={delta:+.1f}s ({pct:+.1f}%)")
    elif ref_wall or test_wall:
        print(f"  ⚠ MF wall partial (ref={ref_wall}, test={test_wall})")

    # Mode P line semantics depend on the gate
    if args.gate == "iso0":
        # np=1: expect NO [Mode P] lines on test
        ok = (len(test_modep) == 0)
        msg = f"Mode P lines: ref={len(ref_modep)}  test={len(test_modep)}  (expect test=0 at np=1)"
        print(("  ✓ " if ok else "  ✗ ") + msg)
        passes.append(("modep_count", ok))
    elif args.gate in ("iso1", "iso2"):
        ok = (len(test_modep) > 0)
        msg = f"Mode P lines: ref={len(ref_modep)}  test={len(test_modep)}  (expect test>0 at np≥2)"
        print(("  ✓ " if ok else "  ✗ ") + msg)
        passes.append(("modep_count", ok))
        # Verify partition coverage
        if test_modep:
            by_model = {}
            for r, m, a, b, n in test_modep:
                by_model.setdefault((m, n), []).append((r, a, b))
            for (m, n), ranges in list(by_model.items())[:3]:
                ranges_sorted = sorted(ranges, key=lambda x: x[1])
                covered = sum(b - a for r, a, b in ranges_sorted)
                ok_cov = (covered == n) and ranges_sorted[0][1] == 0 and ranges_sorted[-1][2] == n
                marker = "    ✓ " if ok_cov else "    ⚠ "
                print(f"{marker}{m}: ranges {ranges_sorted}  covered={covered}/{n}")
                if not ok_cov:
                    passes.append((f"partition_{m}", False))

    # BIC check (optional)
    if args.report_path_ref or args.report_path_test:
        rp_ref  = args.report_path_ref  or (args.ref  / "iqtree_inner.iqtree" if args.ref.is_dir()  else None)
        rp_test = args.report_path_test or (args.test / "iqtree_inner.iqtree" if args.test.is_dir() else None)
        if rp_ref and rp_test and ref_best:
            bic_ref  = parse_bic_from_iqtree_report(rp_ref,  ref_best)
            bic_test = parse_bic_from_iqtree_report(rp_test, ref_best)
            if bic_ref is not None and bic_test is not None:
                dlt = abs(bic_ref - bic_test)
                ok = dlt <= max(1e-4, abs(bic_ref) * 1e-9)
                msg = f"BIC[{ref_best}] ref={bic_ref}  test={bic_test}  |Δ|={dlt:.3e}"
                print(("  ✓ " if ok else "  ✗ ") + msg)
                passes.append(("bic", ok))

    all_pass = all(p[1] for p in passes)
    print()
    if all_pass:
        print(f"══ PARITY: PASS ({args.gate}) ══")
        return 0
    else:
        failed = [name for name, ok in passes if not ok]
        print(f"══ PARITY: FAIL ({args.gate}) — failed checks: {', '.join(failed)} ══", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
