#!/usr/bin/env python3
"""
Harvest additional per-run data from the HPC scratch profile directories
(Gadi /scratch/$PROJECT or Setonix /scratch/$PAWSEY_PROJECT) and merge it
into the existing ``logs/runs/*.json`` files.

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
# Where the run_mega_profile.sh outputs live. Defaults are chosen for Gadi:
#   /scratch/$PROJECT/$USER/iqtree3/gadi-ci/profiles
# but the tool is fully env-driven. Override with:
#   SCRATCH_DIR=/path/to/iqtree3          (root of the project workspace)
#   PROFILE_ROOT=/path/to/profiles        (where <label>_<jobid>/ dirs live)
#   BENCHMARKS_DIR=/path/to/benchmarks    (raw .fa files)
# When harvesting Setonix data, set SCRATCH_DIR=/scratch/pawseyXXXX/$USER/iqtree3
# and PROFILE_ROOT=$SCRATCH_DIR/setonix-ci/profiles.
_PROJECT_ENV = os.environ.get("PROJECT") or os.environ.get("PAWSEY_PROJECT") or "rc29"
_USER = os.environ.get("USER", "")
_DEFAULT_SCRATCH = f"/scratch/{_PROJECT_ENV}/{_USER}/iqtree3"
SCRATCH = Path(os.environ.get("SCRATCH_DIR", _DEFAULT_SCRATCH))
# Try gadi-ci first, then fall back to setonix-ci.
_default_profile_root = SCRATCH / "gadi-ci" / "profiles"
if not _default_profile_root.is_dir() and (SCRATCH / "setonix-ci" / "profiles").is_dir():
    _default_profile_root = SCRATCH / "setonix-ci" / "profiles"
PROFILE_ROOT = Path(os.environ.get("PROFILE_ROOT", str(_default_profile_root)))
BENCHMARKS = Path(os.environ.get("BENCHMARKS_DIR", str(SCRATCH / "benchmarks")))

# Legacy Setonix fallback SLURM job id used when no canonical <label>_<id>
# directory matches. Gadi jobs are glob-matched by label alone so this is
# only used for historical Setonix data.
SLURM_ID = os.environ.get("SLURM_ID", "41703864")


def profile_dir_for(label: str, slurm_id: str | None = None) -> Path:
    """Locate the profile directory for ``label``.

    Resolution order:
    1. ``<label>_<slurm_id>`` if *slurm_id* is provided and the dir exists.
    2. ``<label>_<SLURM_ID>`` (env-var / module-level default) if it exists.
    3. Newest ``<label>_<N>`` dir where ``<N>`` is a purely numeric job ID
       (sorted by mtime desc).  The strict numeric-suffix requirement prevents
       a label such as ``xlarge_mf_4t`` from accidentally matching
       ``xlarge_mf_4t_sr_166978127`` (a different label sharing a prefix).
    """
    if slurm_id:
        candidate = PROFILE_ROOT / f"{label}_{slurm_id}"
        if candidate.is_dir():
            return candidate
    primary = PROFILE_ROOT / f"{label}_{SLURM_ID}"
    if primary.is_dir():
        return primary
    candidates = sorted(
        (
            p for p in PROFILE_ROOT.glob(f"{label}_*")
            if p.is_dir() and p.name[len(label) + 1:].isdigit()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else primary


def parse_profile_meta(path: Path) -> dict:
    """Parse ``profile_meta.json`` emitted by run_mega_profile.sh."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def parse_env_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def parse_iqtree_log_env(log_path: Path) -> dict:
    """Extract hostname, cpu, date, and thread info from an iqtree *.log file.

    IQ-TREE prints a header like::

        Host:    gadi-cpu-spr-0143.gadi.nci.org.au (AVX512, FMA3, 503 GB RAM)
        Command: ...
        Time:    Fri Apr 24 20:46:46 2026
        Kernel:  AVX+FMA - 1 threads (104 CPU cores detected)

    This is the only reliable source of hostname/cpu for Gadi runs where the
    shell env-capture in the PBS worker silently returned empty strings.
    """
    if not log_path.is_file():
        return {}
    out: dict = {}
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Host:") and not out.get("hostname"):
            # "Host:    hostname.domain (CPU flags, N GB RAM)"
            m = re.match(r"Host:\s+(\S+)", stripped)
            if m:
                out["hostname"] = m.group(1)
            # Extract CPU flag summary from parentheses
            m2 = re.search(r"\(([^)]+)\)", stripped)
            if m2 and not out.get("cpu_flags"):
                out["cpu_flags"] = m2.group(1)
        elif stripped.startswith("Time:") and not out.get("date"):
            m = re.match(r"Time:\s+(.+)", stripped)
            if m:
                out["date"] = m.group(1).strip()
        elif stripped.startswith("Kernel:") and not out.get("cpu"):
            # "Kernel:  AVX+FMA - 1 threads (104 CPU cores detected)"
            m = re.search(r"\((\d+)\s+CPU cores detected\)", stripped)
            if m:
                out["cores"] = int(m.group(1))
            m2 = re.match(r"Kernel:\s+(\S.*?)\s+-\s+\d+\s+thread", stripped)
            if m2:
                out["cpu_flags_iqtree"] = m2.group(1).strip()
    return out


def parse_samples_jsonl(path: Path) -> dict:
    """Summarize samples.jsonl into timeseries + peak + io + numa + per-thread."""
    if not path.is_file():
        return {}
    series: list[dict] = []
    peak_rss = 0
    last: dict = {}
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last = rec
                rss = int(rec.get("rss_kb") or 0)
                peak = int(rec.get("peak_kb") or 0)
                if rss > peak_rss:
                    peak_rss = rss
                if peak > peak_rss:
                    peak_rss = peak
                series.append({
                    "t_s": rec.get("t_s"),
                    "rss_kb": rec.get("rss_kb"),
                    "peak_kb": rec.get("peak_kb"),
                    "vms_kb": rec.get("vms_kb"),
                    "threads_now": rec.get("threads_now"),
                    "voluntary_cs": rec.get("voluntary_cs"),
                    "involuntary_cs": rec.get("involuntary_cs"),
                })
    except OSError:
        return {}
    out: dict = {"memory_timeseries": series}
    if peak_rss:
        out["peak_rss_kb"] = peak_rss
    if last:
        if last.get("io"):
            out["io"] = last["io"]
        if last.get("numa"):
            out["numa"] = last["numa"]
        if last.get("per_thread"):
            out["per_thread"] = last["per_thread"]
    return out


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


def _clean_frame(name: str) -> str:
    """Shorten a C++ or runtime frame name for display in flamegraph/callstack."""
    name = name.strip()
    name = re.sub(r"\s*\[clone [^\]]+\]", "", name)  # remove [clone ._omp_fn.0] etc.
    if re.fullmatch(r"0x[0-9a-f]+", name):
        return "[runtime]"
    if name == "0":
        return "[runtime]"
    return name


def parse_hotspots_to_folded(hotspots_txt: Path, limit: int = 500) -> list[dict]:
    """Synthesize folded-stack entries from the call-tree in a perf report
    ``--children`` output (hotspots.txt).

    ``perf_folded.txt`` is the ideal source, but when it is empty (e.g. the
    stackcollapse pipe was killed) this function reconstructs approximate stacks
    from the hierarchical tree that perf report --children embeds under each
    top-level function.  Counts are derived from the reported percentages and
    the total sample count in the header line.

    The output format is identical to ``parse_folded()``:
        [{"stack": "funcA;funcB;funcC", "count": N}, ...]
    sorted by count descending.
    """
    if not hotspots_txt.is_file():
        return []
    text = hotspots_txt.read_text(errors="replace")

    # Total samples from the perf header ("# Samples: 10M of event")
    total_samples = 0
    m = re.search(r"#\s+Samples:\s+([0-9.]+)([MK]?)\s+of event", text)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == "M":
            val *= 1_000_000
        elif unit == "K":
            val *= 1_000
        total_samples = int(val)
    if not total_samples:
        return []

    # A top-level function entry starts with exactly 4 spaces + percentage
    header_re = re.compile(
        r"^\s{4}([0-9.]+)%\s+([0-9]+)\s+\S+\s+\S+\s+\[\.\]\s+(.+?)\s*$"
    )
    # A child tree entry: leading whitespace + optional "|" + "--XX.XX%--FuncName"
    entry_re = re.compile(r"^( +)(?:\|)?--([0-9.]+)%--(.+?)\s*$")

    stacks: dict[str, int] = {}

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        hm = header_re.match(lines[i])
        if hm:
            root_samples = int(hm.group(2))
            root_fn = _clean_frame(hm.group(3))
            key = root_fn
            stacks[key] = stacks.get(key, 0) + root_samples

            # Track (indent_column, fn) pairs for building nested paths
            depth_stack: list[tuple[int, str]] = []
            i += 1
            while i < len(lines) and not header_re.match(lines[i]):
                em = entry_re.match(lines[i])
                if em:
                    indent = len(em.group(1))
                    child_pct = float(em.group(2))
                    child_fn = _clean_frame(em.group(3))
                    child_count = max(1, int(total_samples * child_pct / 100))

                    # Pop back so we're at the correct parent depth
                    while depth_stack and depth_stack[-1][0] >= indent:
                        depth_stack.pop()

                    ancestors = [root_fn] + [fn for _, fn in depth_stack]
                    path = ";".join(ancestors + [child_fn])
                    stacks[path] = stacks.get(path, 0) + child_count
                    depth_stack.append((indent, child_fn))
                i += 1
            continue
        i += 1

    result = [{"stack": k, "count": v} for k, v in stacks.items()]
    result.sort(key=lambda s: s["count"], reverse=True)
    return result[:limit]


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


def _normalise_metric_keys(metrics: dict) -> dict:
    """Strip perf's ':u' / ':k' mode-suffixes so keys match the older runs."""
    if not isinstance(metrics, dict):
        return {}
    out: dict = {}
    for k, v in metrics.items():
        base = k.split(":", 1)[0] if isinstance(k, str) else k
        out[base] = v
    return out


def _derive_rates(metrics: dict) -> None:
    """Fill in IPC and *-rate fields that the dashboard expects."""
    def g(key):
        v = metrics.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def ratio(num_key, den_key, out_key):
        n, d = g(num_key), g(den_key)
        if n is not None and d and out_key not in metrics:
            metrics[out_key] = n / d

    cycles = g("cycles")
    instructions = g("instructions")
    if cycles and instructions and "IPC" not in metrics:
        metrics["IPC"] = instructions / cycles
    ratio("cache-misses", "cache-references", "cache-miss-rate")
    ratio("branch-misses", "branch-instructions", "branch-miss-rate")
    ratio("L1-dcache-load-misses", "L1-dcache-loads", "L1-dcache-miss-rate")
    ratio("dTLB-load-misses", "dTLB-loads", "dTLB-miss-rate")
    ratio("iTLB-load-misses", "iTLB-loads", "iTLB-miss-rate")
    if cycles:
        fe = g("stalled-cycles-frontend")
        be = g("stalled-cycles-backend")
        if fe is not None and "frontend-stall-rate" not in metrics:
            metrics["frontend-stall-rate"] = fe / cycles
        if be is not None and "backend-stall-rate" not in metrics:
            metrics["backend-stall-rate"] = be / cycles


def enrich_run(run: dict) -> bool:
    """Enrich a single run in place. Returns True if anything changed."""
    label = run.get("label")
    if not label:
        return False

    changed = False
    pdir = profile_dir_for(label, slurm_id=run.get("slurm_id"))
    if not pdir.is_dir():
        return False

    iqtree_report = pdir / "iqtree_run.iqtree"
    parsed = parse_iqtree_report(iqtree_report)

    # ── dataset_info ──────────────────────────────────────────────────────
    dataset_name = (run.get("profile") or {}).get("dataset")
    if dataset_name:
        ds_info = {"file": dataset_name}
        # Try the BENCHMARKS path first (works when running on Setonix or
        # with a full scratch mirror). Falls back silently if not present.
        file_info = dataset_file_info(dataset_name)
        ds_info.update(file_info)
        # Pull parsed fields from the .iqtree report
        for key in ("taxa", "sites", "patterns", "constant_sites",
                    "invariant_sites", "informative_sites", "sequence_type"):
            if key in parsed:
                ds_info[key] = parsed[key]
        # Prefer ground-truth size/sha256 from profile_meta.json over the
        # FASTA character-count estimate in dataset_file_info().
        meta_now = parse_profile_meta(pdir / "profile_meta.json")
        meta_ds = (meta_now.get("dataset") or
                   (meta_now.get("env") or {}).get("dataset") or {})
        if meta_ds.get("size_bytes"):
            ds_info["file_size_bytes"] = meta_ds["size_bytes"]
        if meta_ds.get("sha256"):
            ds_info["sha256"] = meta_ds["sha256"]
        if meta_ds.get("path"):
            ds_info["path"] = meta_ds["path"]
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
    # Gadi's matrix worker writes perf_report.txt (same format as Setonix's
    # hotspots.txt but --no-children).  Use whichever is available.
    hotspots_src = pdir / "hotspots.txt"
    if not hotspots_src.is_file():
        hotspots_src = pdir / "perf_report.txt"
    hs = parse_hotspots(hotspots_src)
    if hs and profile.get("hotspots") != hs:
        profile["hotspots"] = hs
        changed = True
    fs = parse_folded(pdir / "perf_folded.txt")
    if not fs:
        # perf_folded.txt is absent (Gadi) or empty (Setonix w/ killed pipe).
        # Synthesize approximate folded stacks from the call-tree embedded in
        # the perf report --children output.  These are derived from % values
        # so counts are estimates, but they give the flamegraph and callstack
        # visualisations enough data to render.
        fs = parse_hotspots_to_folded(hotspots_src)
    if fs and profile.get("folded_stacks") != fs:
        profile["folded_stacks"] = fs
        changed = True

    if profile != (run.get("profile") or {}):
        run["profile"] = profile

    # ── mega-profile extras (profile_meta.json / env.json / samples.jsonl) ─
    meta = parse_profile_meta(pdir / "profile_meta.json")
    env_extra = parse_env_json(pdir / "env.json")
    # Gadi: env fields are empty because the PBS worker's sh() calls silently
    # failed (PATH issues on compute nodes). Recover from iqtree_run.log header.
    iqtree_env = parse_iqtree_log_env(pdir / "iqtree_run.log")
    if iqtree_env:
        env = dict(run.get("env") or {})
        if not env.get("hostname") and iqtree_env.get("hostname"):
            env["hostname"] = iqtree_env["hostname"]
            changed = True
        if not env.get("date") and iqtree_env.get("date"):
            env["date"] = iqtree_env["date"]
            changed = True
        if not env.get("cores") and iqtree_env.get("cores"):
            env["cores"] = iqtree_env["cores"]
            changed = True
        if iqtree_env.get("cpu_flags") and env.get("cpu") != iqtree_env["cpu_flags"]:
            # Use the IQ-TREE header flags as cpu when lscpu is unavailable.
            if not env.get("cpu"):
                env["cpu"] = iqtree_env["cpu_flags"]
                changed = True
        run["env"] = env
    samples_summary = parse_samples_jsonl(pdir / "samples.jsonl")

    if env_extra:
        env = dict(run.get("env") or {})
        for k, v in env_extra.items():
            if k == "dataset":
                continue  # dataset handled separately
            if env.get(k) != v:
                env[k] = v
                changed = True
        if env != (run.get("env") or {}):
            run["env"] = env

    if meta:
        profile = dict(run.get("profile") or {})
        # profile_meta.json may use one of two layouts:
        #   old: {perf_stat: {metrics, raw_events}, ...}
        #   new (run_mega_profile.sh): {profile: {metrics, perf_cmd,
        #                               peak_rss_kb, memory_timeseries, ...}}
        perf_stat = meta.get("perf_stat") or meta.get("profile") or {}
        metrics = perf_stat.get("metrics") or meta.get("metrics") or {}
        raw_events = perf_stat.get("raw_events") or meta.get("raw_events") or {}
        # perf uses a ':u' suffix for user-only counting. Strip it so the
        # dashboard schema keys line up with the earlier runs.
        metrics = _normalise_metric_keys(metrics)
        raw_events = _normalise_metric_keys(raw_events)
        # Compute derived rates (IPC, miss-rates, stall-rates) from the raw
        # counters, matching what run_profiling.sh wrote for the older runs.
        _derive_rates(metrics)
        if metrics:
            merged = dict(profile.get("metrics") or {})
            merged.update(metrics)
            if merged != profile.get("metrics"):
                profile["metrics"] = merged
                changed = True
        if raw_events and profile.get("raw_events") != raw_events:
            profile["raw_events"] = raw_events
            changed = True
        # perf_cmd / peak_rss_kb may live directly under meta['profile']
        for key in ("perf_cmd", "peak_rss_kb"):
            val = perf_stat.get(key)
            if val and profile.get(key) != val:
                profile[key] = val
                changed = True
        run["profile"] = profile

    if samples_summary:
        profile = dict(run.get("profile") or {})
        for k, v in samples_summary.items():
            if profile.get(k) != v:
                profile[k] = v
                changed = True
        run["profile"] = profile

    return changed


def discover_new_profile_runs() -> int:
    """Create stub run JSON files for profile dirs that don't have one yet.

    Scans ``PROFILE_ROOT`` for ``<label>_<slurm>`` directories. If no
    ``logs/runs/<label>_baseline.json`` exists but ``profile_meta.json``
    does, emit a minimal skeleton so the subsequent enrichment pass can
    populate it.
    """
    if not PROFILE_ROOT.is_dir():
        return 0
    created = 0
    for pdir in sorted(PROFILE_ROOT.iterdir()):
        if not pdir.is_dir():
            continue
        meta_path = pdir / "profile_meta.json"
        if not meta_path.is_file():
            continue
        # directory naming convention: <label>_<slurmid>
        parts = pdir.name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        label = parts[0]
        slurm_id = parts[1]
        target = RUNS_DIR / f"{label}_baseline.json"
        if target.exists():
            continue
        meta = parse_profile_meta(meta_path)
        env = (meta.get("env") or {})
        perf_stat = meta.get("perf_stat") or {}
        threads = env.get("threads") or perf_stat.get("threads")
        dataset_file = (env.get("dataset") or {}).get("file") or meta.get("dataset_file")
        wall_time = perf_stat.get("wall_time_s") or meta.get("wall_time_s") or 0
        stub = {
            "run_id": label,
            "slurm_id": slurm_id,
            "label": label,
            "run_type": "deep_profile",
            "timing": ([{"command": "iqtree", "time_s": float(wall_time)}]
                       if wall_time else []),
            "verify": [],
            "env": {k: v for k, v in env.items() if k != "dataset"},
            "profile": {
                "dataset": dataset_file,
                "threads": threads,
            },
            "summary": {
                "pass": 0, "fail": 0,
                "total_time": float(wall_time or 0),
                "all_pass": True,
            },
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(stub, indent=2) + "\n")
        print(f"[harvest] created stub {target.name} from {pdir.name}")
        created += 1
    return created


def main() -> int:
    if not PROFILE_ROOT.is_dir():
        print(f"[harvest] scratch profiles not visible at {PROFILE_ROOT}")
        return 1
    discover_new_profile_runs()
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
