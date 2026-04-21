#!/usr/bin/env python3
"""
Validate logs/runs/*.json and logs/profiles/*.json against JSON schemas.

Exits non-zero if any record is invalid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except ImportError:
    print("jsonschema not installed. Run: pip install jsonschema", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
SCHEMAS = Path(__file__).resolve().parent / "schemas"


def load_schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text())


def validate_dir(directory: Path, schema: dict, label: str) -> int:
    if not directory.is_dir():
        print(f"[validate] {label}: directory missing ({directory}) — skipped")
        return 0
    v = Draft7Validator(schema)
    errors = 0
    files = sorted(directory.glob("*.json"))
    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"[error] {f.relative_to(ROOT)}: invalid JSON — {e}")
            errors += 1
            continue
        for err in v.iter_errors(data):
            loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
            print(f"[error] {f.relative_to(ROOT)} @ {loc}: {err.message}")
            errors += 1
    print(f"[validate] {label}: {len(files)} files, {errors} errors")
    return errors


def main() -> int:
    run_schema = load_schema("run.schema.json")
    prof_schema = load_schema("profile.schema.json")
    errors = 0
    errors += validate_dir(LOGS / "runs", run_schema, "runs")
    errors += validate_dir(LOGS / "profiles", prof_schema, "profiles")
    if errors:
        print(f"\n[validate] FAILED: {errors} total schema errors")
        return 1
    print("\n[validate] OK: all records valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
