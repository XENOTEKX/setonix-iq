"""
Regression detection for performance metrics.

Loads the committed baselines in logs/runs/*_baseline.json and checks that new
runs of the same (dataset, threads) don't regress significantly.

Current mode: WARN ONLY (tests always pass) until we have enough signal to
set meaningful thresholds. Remove the xfail decorator to enforce.
"""
from __future__ import annotations

import pytest

# Thresholds (relative). A new run that is this much WORSE than baseline fails.
WALL_TIME_MAX_REGRESSION = 0.20   # +20% slower triggers warning
IPC_MIN_REGRESSION = -0.15        # -15% IPC triggers warning


def _baselines(runs):
    """Return dict keyed by (dataset, threads) → baseline run."""
    out = {}
    for r in runs:
        rid = r.get("run_id") or ""
        if "baseline" not in rid:
            continue
        p = r.get("profile") or {}
        hints = r.get("hints") or {}
        ds = p.get("dataset") or hints.get("dataset") or r.get("label", "").split("_")[0]
        threads = p.get("threads") or hints.get("threads")
        if ds and threads is not None:
            out[(ds, str(threads))] = r
    return out


@pytest.mark.xfail(reason="warn-only until baselines stabilize", strict=False)
def test_no_wall_time_regressions(runs_raw):
    base = _baselines(runs_raw)
    regressions = []
    for r in runs_raw:
        rid = r.get("run_id") or ""
        if "baseline" in rid:
            continue
        p = r.get("profile") or {}
        hints = r.get("hints") or {}
        ds = p.get("dataset") or hints.get("dataset")
        threads = p.get("threads") or hints.get("threads")
        key = (ds, str(threads)) if ds and threads is not None else None
        if key and key in base:
            baseline_wall = base[key]["summary"]["total_time"]
            cur_wall = r["summary"]["total_time"]
            if baseline_wall > 0:
                delta = (cur_wall - baseline_wall) / baseline_wall
                if delta > WALL_TIME_MAX_REGRESSION:
                    regressions.append(f"{rid}: wall +{delta:.1%} vs {key}")
    assert not regressions, "regressions:\n  " + "\n  ".join(regressions)
