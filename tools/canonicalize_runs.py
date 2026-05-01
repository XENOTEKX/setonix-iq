#!/usr/bin/env python3.11
"""One-shot migration: mark canonical runs, add build_tag, rename Gadi files
so that the canonical gcc/_sr_gcc_pin Gadi runs occupy the convention-mandated
`Gadi_{Dataset}_{T}T.json` slots while older ICX runs are suffixed with
`_sr_icx` to remain distinguishable.

Conventions established:
- Setonix canonical: `Setonix_{Dataset}_{T}T.json`  build_tag = `smtoff_pin`
- Gadi canonical:    `Gadi_{Dataset}_{T}T.json`     build_tag = `sr_gcc_pin`
- Gadi ICX (kept as faded reference):                build_tag = `sr_icx`
                     filename: `Gadi_{Dataset}_{T}T_sr_icx.json`
                     run_id : `Gadi_{Dataset}_{T}T_sr_icx`
- Setonix SMT-on baseline (faded reference):         build_tag = `baseline_smton`
                     (no rename — files already use distinct `_baseline_smton` suffix)

Run ONCE.  Idempotent — re-running detects already-migrated files.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[1] / "logs" / "runs"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def save(p: Path, d: dict) -> None:
    p.write_text(json.dumps(d, indent=2) + "\n")


def is_canonical(run: dict) -> bool:
    return not run.get("archived", False) and not run.get("non_canonical", False)


def main() -> int:
    if not RUNS_DIR.is_dir():
        print(f"runs dir not found: {RUNS_DIR}", file=sys.stderr)
        return 1

    rename_log: list[tuple[str, str]] = []
    flag_log: list[str] = []

    # ---------------------------------------------------------------
    # 1. Move old Gadi ICX large_modelfinder files aside so the new
    #    gcc files can take the canonical name slot.
    # ---------------------------------------------------------------
    for thr in (1, 4, 8, 16, 32, 64, 104):
        old_icx = RUNS_DIR / f"Gadi_large_modelfinder_{thr}T.json"
        new_icx = RUNS_DIR / f"Gadi_large_modelfinder_{thr}T_sr_icx.json"
        if old_icx.exists() and not new_icx.exists():
            d = load(old_icx)
            label = (d.get("label") or "").lower()
            if "_sr_icx" in label or d.get("non_canonical"):
                d["run_id"] = f"Gadi_large_modelfinder_{thr}T_sr_icx"
                d["build_tag"] = "sr_icx"
                save(new_icx, d)
                old_icx.unlink()
                rename_log.append((old_icx.name, new_icx.name))

    # ---------------------------------------------------------------
    # 2. Promote new Gadi gcc canonical files to canonical slots.
    # ---------------------------------------------------------------
    for thr in (1, 4, 8, 16, 32, 64, 104):
        old_gcc = RUNS_DIR / f"gadi_large_modelfinder_{thr}t_sr_gcc_pin.json"
        new_gcc = RUNS_DIR / f"Gadi_large_modelfinder_{thr}T.json"
        if old_gcc.exists() and not new_gcc.exists():
            d = load(old_gcc)
            d["run_id"] = f"Gadi_large_modelfinder_{thr}T"
            d["canonical"] = True
            d["build_tag"] = "sr_gcc_pin"
            save(new_gcc, d)
            old_gcc.unlink()
            rename_log.append((old_gcc.name, new_gcc.name))

    # ---------------------------------------------------------------
    # 3. Tag remaining Gadi ICX files (xlarge_mf, mega_dna).  No rename
    #    needed because no gcc replacement exists yet for those datasets.
    # ---------------------------------------------------------------
    for p in sorted(RUNS_DIR.glob("Gadi_xlarge_mf_*.json")) + sorted(RUNS_DIR.glob("Gadi_mega_dna_*.json")):
        if p.name.endswith("_sr_icx.json"):
            continue
        d = load(p)
        if d.get("non_canonical") and not d.get("build_tag"):
            d["build_tag"] = "sr_icx"
            save(p, d)
            flag_log.append(f"{p.name}: build_tag=sr_icx")

    # ---------------------------------------------------------------
    # 4. Mark Setonix canonical runs.
    # ---------------------------------------------------------------
    for p in sorted(RUNS_DIR.glob("Setonix_*.json")):
        d = load(p)
        if not is_canonical(d):
            continue
        changed = False
        if d.get("canonical") is not True:
            d["canonical"] = True
            changed = True
        if not d.get("build_tag"):
            d["build_tag"] = "smtoff_pin"
            changed = True
        if changed:
            save(p, d)
            flag_log.append(f"{p.name}: canonical=true, build_tag=smtoff_pin")

    # ---------------------------------------------------------------
    # 5. Tag Setonix SMT-on baseline runs.
    # ---------------------------------------------------------------
    for p in sorted(RUNS_DIR.glob("*_baseline_smton.json")):
        d = load(p)
        if not d.get("non_canonical"):
            continue
        if not d.get("build_tag"):
            d["build_tag"] = "baseline_smton"
            save(p, d)
            flag_log.append(f"{p.name}: build_tag=baseline_smton")

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    print("=" * 60)
    print(f"Renames: {len(rename_log)}")
    for o, n in rename_log:
        print(f"  {o}  →  {n}")
    print()
    print(f"Flag updates: {len(flag_log)}")
    for line in flag_log:
        print(f"  {line}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
