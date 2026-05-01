#!/usr/bin/env python3.11
"""One-shot migration (follow-up #17): convert legacy ratio-format `*-rate`
metric fields (0-1) in `logs/runs/*.json` and `logs/profiles/*.json` to
percent (0-100), matching the Gadi worker convention and the dashboard's
fmtPercent() / radar normalize() assumptions.

A run is considered "ratio-format" if its `cache-miss-rate` (or any *-miss-rate
field) has a value that is plausibly a ratio (i.e. < 1.0 AND not NaN). Files
that already contain percent-format values (>= 1.0) are left untouched —
this keeps the migration idempotent.

Affected fields (all rescaled by *100, rounded to 4 decimals):
  cache-miss-rate, l2-miss-rate, l3-miss-rate
  branch-miss-rate
  L1-dcache-miss-rate
  dTLB-miss-rate, iTLB-miss-rate
  frontend-stall-rate, backend-stall-rate
  l3-prefetch-miss-rate

NOT affected (not percent-typed):
  IPC, *-mpki, *-mpki, cache_level, cycles, instructions, etc.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PERCENT_FIELDS = (
    "cache-miss-rate",
    "l2-miss-rate",
    "l3-miss-rate",
    "branch-miss-rate",
    "L1-dcache-miss-rate",
    "dTLB-miss-rate",
    "iTLB-miss-rate",
    "frontend-stall-rate",
    "backend-stall-rate",
    "l3-prefetch-miss-rate",
)


def is_ratio_format(metrics: dict) -> bool:
    """A file is ratio-format if its cache-miss-rate is < 1.0.

    cache-miss-rate is essentially never < 1 % in real HPC workloads (would
    mean the cache absorbs > 99 % of references), so a value < 1 unambiguously
    identifies a ratio-format value. Fall back to L1-dcache-miss-rate or
    branch-miss-rate when cache-miss-rate is absent.
    """
    for key in ("cache-miss-rate", "L1-dcache-miss-rate", "branch-miss-rate"):
        v = metrics.get(key)
        if isinstance(v, (int, float)) and v == v:  # not NaN
            return v < 1.0
    return False


def migrate_metrics(metrics: dict) -> int:
    """Rescale percent-typed fields in-place. Returns count of fields touched."""
    n = 0
    for key in PERCENT_FIELDS:
        v = metrics.get(key)
        if isinstance(v, (int, float)) and v == v:
            metrics[key] = round(100.0 * v, 4)
            n += 1
    return n


def migrate_file(path: Path, dry_run: bool) -> tuple[bool, int]:
    """Migrate one run/profile JSON. Returns (changed, n_fields_touched)."""
    try:
        rec = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  SKIP {path.name}: {e}", file=sys.stderr)
        return False, 0

    metrics = (rec.get("profile") or {}).get("metrics")
    if not isinstance(metrics, dict):
        return False, 0
    if not is_ratio_format(metrics):
        return False, 0

    n = migrate_metrics(metrics)
    if n == 0:
        return False, 0
    if not dry_run:
        path.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    return True, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing files")
    ap.add_argument("--root", default=".",
                    help="repo root (defaults to cwd)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    targets = sorted(
        (root / "logs" / "runs").glob("*.json")
    ) + sorted(
        (root / "logs" / "profiles").glob("*.json")
    )

    if not targets:
        print(f"No JSON files found under {root}/logs/{{runs,profiles}}/")
        return 1

    n_changed = 0
    n_unchanged = 0
    n_fields = 0
    for p in targets:
        changed, fields = migrate_file(p, args.dry_run)
        if changed:
            n_changed += 1
            n_fields += fields
            print(f"  {'WOULD-MIGRATE' if args.dry_run else 'MIGRATED'}  {p.name}  ({fields} fields)")
        else:
            n_unchanged += 1

    print()
    print(f"[migrate] {len(targets)} files scanned, "
          f"{n_changed} {'would be ' if args.dry_run else ''}migrated "
          f"({n_fields} field rescales), {n_unchanged} unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
