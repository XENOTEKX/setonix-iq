"""Extra schema checks specific to the gadi-iq branch.

Verifies that PBS-shaped records (as emitted by gadi-ci/run_mega_profile.sh
and gadi-ci/run_pipeline.sh) validate against the unified run schema without
requiring any SLURM/Pawsey/AMD fields.
"""
import json
from pathlib import Path

from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = Draft7Validator(
    json.loads((ROOT / "tools" / "schemas" / "run.schema.json").read_text())
)


def _minimal_gadi_run() -> dict:
    return {
        "run_id": "2026-04-24_120000",
        "pbs_id": "12345678",
        "run_type": "pipeline",
        "label": "gadi_pipeline_test",
        "timing": [{"command": "iqtree3 -s turtle.fa", "time_s": 1.62}],
        "verify": [{"file": "turtle.fa", "status": "pass",
                    "expected": -5681.1, "reported": -5681.1, "diff": 0.0}],
        "env": {
            "date": "2026-04-24T12:00:00+11:00",
            "hostname": "gadi-cpu-clx-3021",
            "cpu": "Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz",
            "cores": 48,
            "gcc": "8.5.0",
            "kernel": "4.18.0-553.92.1.el8.nci.x86_64",
            "os": "Rocky Linux 8.10 (Green Obsidian)",
            "vtune_version": "Intel(R) VTune(TM) Profiler 2024.2.0",
            "pbs": {
                "job_id": "12345678.gadi-pbs",
                "job_name": "iqtree-mega-48t",
                "queue": "normal",
                "project": "rc29",
                "ncpus": "48",
                "submit_host": "gadi-login-03",
                "submit_dir": "/scratch/rc29/as1708/iqtree3/gadi-ci",
                "scheduler": "pbs_pro",
            },
        },
        "summary": {"pass": 1, "fail": 0, "total_time": 1.62, "all_pass": True},
    }


def test_minimal_gadi_record_validates():
    run = _minimal_gadi_run()
    errors = list(SCHEMA.iter_errors(run))
    assert not errors, "unexpected schema errors:\n  " + "\n  ".join(
        e.message for e in errors
    )


def test_gadi_record_with_intel_tma_and_vtune_validates():
    run = _minimal_gadi_run()
    run["profile"] = {
        "dataset": "mega_dna.fa",
        "threads": 48,
        "perf_cmd": "perf stat -e cycles,instructions,... -- iqtree3 ...",
        "metrics": {
            "IPC": 1.42,
            "LLC-miss-rate": 1.18,
            "intel-tma-retiring-pct": 28.4,
            "intel-tma-bad-spec-pct": 3.1,
            "intel-tma-frontend-bound-pct": 41.2,
            "intel-tma-backend-bound-pct": 27.3,
        },
        "vtune": {
            "elapsed_time_s": 120.5,
            "cpu_time_s": 5424.0,
            "effective_cpu_util": 88.1,
            "avg_cpu_freq_ghz": 2.9,
            "hotspots": [
                {"function": "computePartialLikelihoodSIMD",
                 "module": "iqtree3", "cpu_time_s": 2340.2},
            ],
        },
    }
    errors = list(SCHEMA.iter_errors(run))
    assert not errors, "unexpected schema errors:\n  " + "\n  ".join(
        e.message for e in errors
    )


def test_setonix_legacy_record_still_validates():
    """SLURM/Pawsey/AMD fields must remain accepted alongside PBS/Intel."""
    run = _minimal_gadi_run()
    del run["pbs_id"]
    run["slurm_id"] = "41849113"
    run["env"].pop("pbs", None)
    run["env"]["rocm"] = "6.3.0"
    run["env"]["slurm"] = {
        "job_id": "41849113",
        "job_name": "iqtree-mega-128t",
        "partition": "work",
        "account": "pawsey1351",
        "cpus_per_task": "128",
    }
    errors = list(SCHEMA.iter_errors(run))
    assert not errors, "unexpected schema errors:\n  " + "\n  ".join(
        e.message for e in errors
    )
