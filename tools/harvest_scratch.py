#!/usr/bin/env python3
"""
Harvest additional per-run data from the Setonix scratch profile directories
and merge it into the existing ``logs/runs/*.json`` files.

Fills the following fields (all optional, schema-compliant):

- ``dataset_info``        — taxa, sites, patterns, informative/constant sites,
                            file size, sequence type (from the ``.iqtree``
                            report + the raw ``.fa`` file if present).
- ``modelfinder``         — adds ``log_likelihood``, ``bic``, ``aic``, ``aicc``,
                            ``tree_length``, ``gamma_alpha``, ``candidates[]``
                            (top-10 from the BIC-sorted table).
- ``profile.perf_cmd``    — the exact perf command (parsed from the
                            ``perf_stat.txt`` header).
- ``profile.hotspots[]``  — parsed from ``hotspots.txt`` when ``perf record`` was
                            run (only 1T/64T/128T under the current pipeline).
- ``profile.folded_stacks[]`` — parsed from ``perf_folded.txt`` (same runs).

The script is idempotent: re-running it refreshes each field from the source
of truth.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "logs" / "runs"
SCRATCH = Path("/scratch/pawsey1351/asamuel/iqtree3")
PROFILE_ROOT = SCRATCH / "setonix-ci" / "profiles"
BENCHMARKS = SCRATCH / "benchmarks"

# Map the run labels we have in logs/runs to scratch profile directory names.
# Scratch dirs are suffixed with the SLURM ID of the parent job (41703864).
SLURM_ID = "41703864"


def profile_dir_for(label: str) -> Path:
    return PROFILE_ROOT / f"{label}_{SLURM_ID}"


def parse_iqtree_report(path: Path) -> dict:
    """Pull dataset + model metadata out of an ``*.iqtree`` analysis report."""
    if not path.is_file():
        return {}
    text = path.read_text(errors="replace")

    info: dict = {}

    m = re.search(r"Input data:\s+(\d+)\s+\w+\s+with\s+(\d+)\s+(\w+)\s+sites", text)
    if m:
        info["taxa"] = int(m.group(1))
        info["sites"] = int(m.group(2))
        kind = m.group(3).lower()
        if "nucleotide" in kind:
            info["sequence_type"] = "DNA"
        elif "amino" in kind:
            info["sequence_type"] = "AA"
        elif "codon" in kind:
            info["sequence_type"] = "codon"
        else:
            info["sequence_type"] = kind

    m = re.search(r"Number of distinct site patterns:\s+(\d+)", text)
    if m:
        info["patterns"] = int(m.group(1))
    m = re.search(r"Number of constant sites:\s+(\d+)", text)
    if m:
        info["constant_sites"] = int(m.group(1))
    m = re.search(r"Number of invariant \(constant or ambiguous constant\) sites:\s+(\d+)", text)
    if m:
        info["invariant_sites"] = int(m.group(1))
    m = re.search(r"Number of parsimony informative sites:\s+(\d+)", text)
    if m:
        info["informative_sites"] = int(m.group(1))

    # Model & tree stats
    model_info: dict = {}
    m = re.search(r"Best-fit model according to BIC:\s+(\S+)", text)
    if m:
        model_info["best_model_bic"] = m.group(1)
    m = re.search(r"Model of substitution:\s+(\S+)", text)
    if m:
        model_info["model_selected"] = m.group(1)
    m = re.search(r"Log-likelihood of the tree:\s+([-0-9.]+)", text)
    if m:
        model_info["log_likelihood"] = float(m.group(1))
    m = re.search(r"Unconstrained log-likelihood.*?:\s+([-0-9.]+)", text)
    if m:
        model_info["unconstrained_log_likelihood"] = float(m.group(1))
    m = re.search(r"Gamma shape alpha:\s+([-0-9.]+)", text)
    if m:
        model_info["gamma_alpha"] = float(m.group(1))
    m = re.search(r"Total tree length \(sum of branch lengths\):\s+([-0-9.]+)", text)
    if m:
        model_info["tree_length"] = float(m.group(1))
    m = re.search(r"Bayesian information criterion \(BIC\) score:\s+([-0-9.]+)", text)
    if m:
        model_info["bic"] = float(m.group(1))
    m = re.search(r"Akaike information criterion \(AIC\) score:\s+([-0-9.]+)", text)
    if m:
        model_info["aic"] = float(m.group(1))
    m = re.search(r"Corrected Akaike information criterion \(AICc\) score:\s+([-0-9.]+)", text)
    if m:
        model_info["aicc"] = float(m.group(1))

    # Top-10 model candidates from the BIC-sorted table
    candidates = []
    candidate_re = re.compile(
        r"^(\S+)\s+(-?[0-9.]+)\s+([0-9.]+)\s+([+\-])\s+([0-9e.+\-]+)\s+"
        r"([0-9.]+)\s+([+\-])\s+([0-9e.+\-]+)\s+([0-9.]+)\s+([+\-])\s+([0-9e.+\-]+)\s*$"
    )
    in_list = False
    for line in text.splitlines():
        if line.startswith("List of models sorted by BIC scores"):
            in_list = True
            continue
        if in_list:
            stripped = line.strip()
            if not stripped:
                # blank inside or after the block
                if candidates:
                    break
                continue
            if stripped.startswith("Model") or stripped.startswith("AIC,"):
                continue
            m = candidate_re.match(line)
            if m:
                candidates.append({
                    "model": m.group(1),
                    "log_likelihood": float(m.group(2)),
                    "aic": float(m.group(3)),
                    "aic_weight": _weight(m.group(4), m.group(5)),
                    "aicc": float(m.group(6)),
                    "aicc_weight": _weight(m.group(7), m.group(8)),
                    "bic": float(m.group(9)),
                    "bic_weight": _weight(m.group(10), m.group(11)),
                })
                if len(candidates) >= 10:
                    break
            else:
                # hit something that isn't a candidate row after having some
                if candidates:
                    break
    if candidates:
        model_info["candidates"] = candidates

    if model_info:
        info["_modelfinder"] = model_info
    return info


def _weight(sign: str, val: str) -> float:
    # The +/- in IQ-TREE's candidate table is a significance indicator, not a
    # sign; the weight itself is always non-negative.
    try:
        return abs(float(val))
    except ValueError:
        return 0.0


def parse_perf_cmd(perf_stat_txt: Path) -> str | None:
    if not perf_stat_txt.is_file():
        return None
    for line in perf_stat_txt.read_text(errors="replace").splitlines():
        line = line.strip()
        m = re.match(r"Performance counter stats for '(.*)':", line)
        if m:
            return m.group(1)
    return None


def parse_hotspots(hotspots_txt: Path, limit: int = 15) -> list[dict]:
    """Parse ``perf report --stdio --no-children`` output into hotspot records."""
    if not hotspots_txt.is_file():
        return []
    out = []
    # lines look like:
    #   36.77%  35004976  iqtree3  iqtree3  [.] function-signature
    row_re = re.compile(
        r"^\s*([0-9.]+)%\s+([0-9]+)\s+(\S+)\s+(\S+)\s+\[\.\]\s+(.+?)\s*$"
    )
    for line in hotspots_txt.read_text(errors="replace").splitlines():
        m = row_re.match(line)
        if not m:
            continue
        out.append({
            "percent": float(m.group(1)),
            "samples": int(m.group(2)),
            "command": m.group(3),
            "module": m.group(4),
            "function": m.group(5),
        })
        if len(out) >= limit:
            break
    return out


def parse_folded(folded_txt: Path, limit: int | None = None) -> list[dict]:
    if not folded_txt.is_file():
        return []
    out = []
    for line in folded_txt.read_text(errors="replace").splitlines():
        line = line.rstrip()
        if not line:
            continue
        # "stack;stack;stack COUNT"
        idx = line.rfind(" ")
        if idx <= 0:
            continue
        try:
            count = int(line[idx + 1:].strip())
        except ValueError:
            continue
        stack = line[:idx]
        out.append({"stack": stack, "count": count})
    # sort by count descending; preserve full list unless a limit is requested
    out.sort(key=lambda r: r["count"], reverse=True)
    if limit is not None:
        out = out[:limit]
    return out


def dataset_file_info(filename: str) -> dict:
    """Measure actual taxa / sites / size of a FASTA on scratch."""
    path = BENCHMARKS / filename
    if not path.is_file():
        return {}
    info = {"file": filename, "file_size_bytes": path.stat().st_size}
    taxa = 0
    first_seq_len = 0
    first_seq_collected = False
    try:
        with path.open() as f:
            for line in f:
                if line.startswith(">"):
                    taxa += 1
                    if taxa == 2:
                        first_seq_collected = True
                elif not first_seq_collected:
                    first_seq_len += len(line.strip())
    except OSError:
        return info
    if taxa:
        info["fasta_taxa"] = taxa
        if first_seq_len:
            info["fasta_sites"] = first_seq_len
    return info


def enrich_run(run: dict) -> bool:
    """Enrich a single run in place. Returns True if anything changed."""
    label = run.get("label")
    if not label:
        return False

    changed = False
    pdir = profile_dir_for(label)
    if not pdir.is_dir():
        return False

    iqtree_report = pdir / "iqtree_run.iqtree"
    parsed = parse_iqtree_report(iqtree_report)

    # ── dataset_info ──────────────────────────────────────────────────────
    dataset_name = (run.get("profile") or {}).get("dataset")
    if dataset_name:
        ds_info = {"file": dataset_name}
        file_info = dataset_file_info(dataset_name)
        ds_info.update(file_info)
        for key in ("taxa", "sites", "patterns", "constant_sites",
                    "invariant_sites", "informative_sites", "sequence_type"):
            if key in parsed:
                ds_info[key] = parsed[key]
        if ds_info != run.get("dataset_info"):
            run["dataset_info"] = ds_info
            changed = True

    # ── modelfinder (enrich / merge) ──────────────────────────────────────
    extra_mf = parsed.get("_modelfinder") or {}
    if extra_mf:
        mf = dict(run.get("modelfinder") or {})
        for k, v in extra_mf.items():
            if mf.get(k) != v:
                mf[k] = v
                changed = True
        if mf != run.get("modelfinder"):
            run["modelfinder"] = mf
            changed = True

    # ── profile.perf_cmd ──────────────────────────────────────────────────
    profile = dict(run.get("profile") or {})
    perf_cmd = parse_perf_cmd(pdir / "perf_stat.txt")
    if perf_cmd and profile.get("perf_cmd") != perf_cmd:
        profile["perf_cmd"] = perf_cmd
        changed = True

    # ── profile.hotspots / folded_stacks (only if raw data exists) ────────
    hs = parse_hotspots(pdir / "hotspots.txt")
    if hs and profile.get("hotspots") != hs:
        profile["hotspots"] = hs
        changed = True
    fs = parse_folded(pdir / "perf_folded.txt")
    if fs and profile.get("folded_stacks") != fs:
        profile["folded_stacks"] = fs
        changed = True

    if profile != (run.get("profile") or {}):
        run["profile"] = profile

    return changed


def main() -> int:
    if not PROFILE_ROOT.is_dir():
        print(f"[harvest] scratch profiles not visible at {PROFILE_ROOT}")
        return 1
    touched = 0
    for run_file in sorted(RUNS_DIR.glob("*.json")):
        try:
            run = json.loads(run_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[harvest] skip {run_file.name}: {e}")
            continue
        if enrich_run(run):
            run_file.write_text(json.dumps(run, indent=2) + "\n")
            print(f"[harvest] updated {run_file.name}")
            touched += 1
    print(f"[harvest] done — {touched} run file(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
