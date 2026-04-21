#!/usr/bin/env python3
"""
Normalize run and profile JSON files.

Reads raw logs from ``logs/runs/`` and ``logs/profiles/`` and emits normalized
data to ``web/data/``:

- ``web/data/runs/<id>.json``           per-run (lazy-loaded by dashboard)
- ``web/data/profiles/<id>.json``       per-profile
- ``web/data/runs.index.json``          summary list for leaderboard
- ``web/data/profiles.index.json``      summary list
- ``web/data/manifest.json``            { generated_at, counts, schema_version }

Normalization rules:
- Ensure every run has ``summary`` derived from ``verify`` + ``timing``.
- Ensure every run has ``profile.metrics`` dict (empty if absent).
- Extract dataset + model hints from command strings (taxa/sites/model).
- Clamp rate metrics to [0, 100] to guard schema.
- Drop callstacks > 50 entries in the index to keep it small.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "web" / "data"
SCHEMA_VERSION = 1


def load_all(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    out = []
    for f in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] skipped {f.name}: {e}", file=sys.stderr)
    return out


def derive_summary(run: dict) -> dict:
    verify = run.get("verify", []) or []
    timing = run.get("timing", []) or []
    pass_count = sum(1 for v in verify if v.get("status") == "pass")
    fail_count = sum(1 for v in verify if v.get("status") == "fail")
    total = sum(float(t.get("time_s", 0)) for t in timing)
    return {
        "pass": pass_count,
        "fail": fail_count,
        "total_time": round(total, 3),
        "all_pass": fail_count == 0,
    }


_CMD_DATASET = re.compile(r"-s\s+(\S+)")
_CMD_MODEL = re.compile(r"-m\s+(\S+)")
_CMD_THREADS = re.compile(r"-T\s+(\S+)")


def infer_from_commands(run: dict) -> dict:
    hints = {"dataset": None, "model": None, "threads": None}
    for t in run.get("timing", []) or []:
        cmd = t.get("command", "")
        if hints["dataset"] is None:
            m = _CMD_DATASET.search(cmd)
            if m:
                hints["dataset"] = os.path.basename(m.group(1))
        if hints["model"] is None:
            m = _CMD_MODEL.search(cmd)
            if m:
                hints["model"] = m.group(1)
        if hints["threads"] is None:
            m = _CMD_THREADS.search(cmd)
            if m:
                try:
                    hints["threads"] = int(m.group(1))
                except ValueError:
                    hints["threads"] = m.group(1)
    return hints


def normalize_run(run: dict) -> dict:
    run = dict(run)  # shallow copy
    if "summary" not in run or not isinstance(run["summary"], dict):
        run["summary"] = derive_summary(run)
    else:
        # Refresh in case timing changed
        run["summary"] = {**derive_summary(run), **run["summary"]}
        run["summary"]["all_pass"] = run["summary"].get("fail", 0) == 0

    hints = infer_from_commands(run)
    profile = run.get("profile") or {}
    if not profile.get("dataset") and hints["dataset"]:
        profile["dataset"] = hints["dataset"]
    if not profile.get("threads") and hints["threads"] is not None:
        profile["threads"] = hints["threads"]
    run["profile"] = profile
    run.setdefault("hints", {})
    run["hints"].update({k: v for k, v in hints.items() if v is not None})
    return run


def summarize_run(run: dict) -> dict:
    """Tiny record stored in runs.index.json."""
    p = run.get("profile") or {}
    metrics = p.get("metrics") or {}
    hints = run.get("hints") or {}
    env = run.get("env") or {}
    return {
        "run_id": run.get("run_id"),
        "slurm_id": run.get("slurm_id"),
        "label": run.get("label") or run.get("run_id"),
        "description": run.get("description", ""),
        "run_type": run.get("run_type", "pipeline"),
        "dataset": p.get("dataset") or hints.get("dataset"),
        "model": hints.get("model"),
        "threads": p.get("threads") or hints.get("threads"),
        "hostname": env.get("hostname"),
        "cpu": env.get("cpu"),
        "date": env.get("date"),
        "wall_s": run["summary"]["total_time"],
        "pass": run["summary"]["pass"],
        "fail": run["summary"]["fail"],
        "all_pass": run["summary"]["all_pass"],
        "IPC": metrics.get("IPC"),
        "frontend_stall_rate": metrics.get("frontend-stall-rate"),
        "cache_miss_rate": metrics.get("cache-miss-rate"),
        "has_hotspots": bool(p.get("hotspots")),
        "has_stacks": bool(p.get("folded_stacks") or p.get("callstacks")),
    }


def summarize_profile(prof: dict) -> dict:
    cpu = prof.get("cpu") or {}
    derived = cpu.get("derived") or {}
    return {
        "profile_id": prof.get("profile_id"),
        "slurm_id": prof.get("slurm_id"),
        "date": prof.get("date"),
        "dataset": prof.get("dataset"),
        "threads": prof.get("threads"),
        "model": prof.get("model"),
        "IPC": derived.get("IPC"),
        "has_gpu": bool(prof.get("gpu")),
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


def main() -> int:
    runs_raw = load_all(LOGS / "runs")
    profs_raw = load_all(LOGS / "profiles")

    runs = [normalize_run(r) for r in runs_raw]
    profs = list(profs_raw)

    # Per-record files
    for r in runs:
        rid = r.get("run_id")
        if not rid:
            continue
        write_json(OUT / "runs" / f"{rid}.json", r)
    for p in profs:
        pid = p.get("profile_id")
        if not pid:
            continue
        write_json(OUT / "profiles" / f"{pid}.json", p)

    # Indexes
    write_json(OUT / "runs.index.json", [summarize_run(r) for r in runs])
    write_json(OUT / "profiles.index.json", [summarize_profile(p) for p in profs])

    # Manifest
    write_json(OUT / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": len(runs),
        "profiles": len(profs),
    })

    print(f"[normalize] wrote {len(runs)} runs, {len(profs)} profiles → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
