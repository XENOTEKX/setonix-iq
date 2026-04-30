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


# Curated dataset profiles. The HPC scripts don't embed taxa/site counts in the
# run JSON, so we keep a static lookup here. `size_mb` is computed from FASTA
# geometry (taxa × sites + header overhead). Sizes are marked estimated because
# the raw .fa files are not bundled with this repo.
# NOTE: values verified against /scratch/pawsey1351/asamuel/iqtree3/benchmarks
# on 2026-04-21. Use `dataset_info` on the run record itself when available;
# this lookup is only a fallback for runs that predate the harvest step.
DATASET_INFO: dict[str, dict] = {
    "turtle.fa":            {"taxa": 16,  "sites": 20820,  "kind": "dna"},
    "large_modelfinder.fa": {"taxa": 100, "sites": 50000,  "kind": "dna"},
    "xlarge_dna.fa":        {"taxa": 200, "sites": 100000, "kind": "dna"},
    "medium_dna.fa":        {"taxa": 50,  "sites": 4559,   "kind": "dna"},
    "example.phy":          {"taxa": 17,  "sites": 1998,   "kind": "dna"},
    "mega_dna.fa":          {"taxa": 500, "sites": 100000, "kind": "dna"},
}


def _compute_size_mb(info: dict) -> float | None:
    taxa = info.get("taxa")
    sites = info.get("sites")
    if not taxa or not sites:
        return None
    # FASTA: ">hdr\n" (~32B avg) + sequence (1B/site for DNA) + "\n"
    bytes_est = taxa * (32 + sites + 1)
    return round(bytes_est / 1_000_000, 2)


def dataset_lookup(name: str | None) -> dict | None:
    if not name:
        return None
    base = os.path.basename(name)
    info = DATASET_INFO.get(base) or DATASET_INFO.get(name)
    if not info:
        return None
    enriched = dict(info)
    enriched["size_mb"] = _compute_size_mb(info)
    enriched["size_estimated"] = True
    return enriched


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

    # Platform detection: explicit field wins, otherwise infer from scheduler id.
    if not run.get("platform"):
        if run.get("pbs_id"):
            run["platform"] = "gadi"
        elif run.get("slurm_id"):
            run["platform"] = "setonix"

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
    mf = run.get("modelfinder") or {}
    dataset = p.get("dataset") or hints.get("dataset")

    # Prefer harvested ground truth over the heuristic lookup.
    # Gadi sparse records only carry fasta_taxa/fasta_sites (no alisim-parsed
    # counts) — fall back to those before the curated lookup.
    ds_gt = run.get("dataset_info") or {}
    fallback = dataset_lookup(dataset) or {}
    taxa = ds_gt.get("taxa") or ds_gt.get("fasta_taxa") or fallback.get("taxa")
    sites = ds_gt.get("sites") or ds_gt.get("fasta_sites") or fallback.get("sites")
    patterns = ds_gt.get("patterns") or ds_gt.get("fasta_patterns")
    if ds_gt.get("file_size_bytes"):
        size_mb = round(ds_gt["file_size_bytes"] / 1_000_000, 2)
        size_estimated = False
    else:
        size_mb = fallback.get("size_mb")
        size_estimated = fallback.get("size_estimated", False)

    return {
        "run_id": run.get("run_id"),
        "slurm_id": run.get("slurm_id"),
        "pbs_id": run.get("pbs_id"),
        "platform": run.get("platform"),
        "label": run.get("label") or run.get("run_id"),
        "description": run.get("description", ""),
        "run_type": run.get("run_type", "pipeline"),
        "dataset": dataset,
        "dataset_short": os.path.basename(dataset) if dataset else None,
        "taxa": taxa,
        "sites": sites,
        "patterns": patterns,
        "informative_sites": ds_gt.get("informative_sites"),
        "constant_sites": ds_gt.get("constant_sites"),
        "sequence_type": ds_gt.get("sequence_type"),
        "size_mb": size_mb,
        "size_estimated": size_estimated,
        "model": mf.get("model_selected") or hints.get("model"),
        "model_bic": mf.get("bic"),
        "model_aic": mf.get("aic"),
        "gamma_alpha": mf.get("gamma_alpha"),
        "tree_length": mf.get("tree_length"),
        "log_likelihood": mf.get("log_likelihood"),
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
        "has_candidates": bool(mf.get("candidates")),
        "has_perf_cmd": bool(p.get("perf_cmd")),
        "dataset_canonical": ds_gt.get("dataset_canonical"),
        "dataset_canonical_note": ds_gt.get("dataset_canonical_note"),
        "archived": run.get("archived", False),
    }


def enrich_index_with_speedup(index: list[dict]) -> None:
    """Add ``speedup`` and ``efficiency`` per entry using the 1-thread baseline
    for the *same dataset on the same platform*.

    Keying by ``(dataset_short, platform)`` prevents cross-platform contamination
    where, for example, a Setonix 1T baseline for ``large_modelfinder.fa`` would
    inflate a Gadi 16T run's speedup (the two platforms regenerate datasets with
    different dimensions).
    """
    baseline: dict[tuple[str | None, str | None], float] = {}
    for r in index:
        ds = r.get("dataset_short")
        plat = r.get("platform")
        # Ignore failed / stub runs when choosing a baseline.
        if (ds and r.get("threads") == 1
                and r.get("wall_s")
                and r.get("all_pass")):
            baseline.setdefault((ds, plat), r["wall_s"])
    for r in index:
        ds = r.get("dataset_short")
        plat = r.get("platform")
        t = r.get("threads") or 0
        w = r.get("wall_s") or 0
        base = baseline.get((ds, plat))
        if base and w and r.get("all_pass"):
            r["speedup"] = round(base / w, 3)
            r["efficiency"] = round((base / w) / t, 4) if t else None
        else:
            r["speedup"] = None
            r["efficiency"] = None


def summarize_profile(prof: dict) -> dict:
    cpu = prof.get("cpu") or {}
    derived = cpu.get("derived") or {}
    return {
        "profile_id": prof.get("profile_id"),
        "slurm_id": prof.get("slurm_id"),
        "pbs_id": prof.get("pbs_id"),
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
    runs_index = [summarize_run(r) for r in runs]
    enrich_index_with_speedup(runs_index)
    write_json(OUT / "runs.index.json", runs_index)
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
