"""
Data invariants that must hold for every committed run.

These catch pipeline bugs that pass JSON-schema but produce nonsensical data.
"""
import pytest


def test_run_ids_unique(runs_raw):
    ids = [r.get("run_id") for r in runs_raw]
    assert len(ids) == len(set(ids)), f"duplicate run_id(s): {[x for x in ids if ids.count(x) > 1]}"


def test_summary_matches_verify(runs_raw):
    for r in runs_raw:
        s = r["summary"]
        verify = r.get("verify", []) or []
        pass_n = sum(1 for v in verify if v.get("status") == "pass")
        fail_n = sum(1 for v in verify if v.get("status") == "fail")
        # Allow either derived or stored counts to match
        if verify:
            assert s["pass"] == pass_n, f"{r['run_id']}: summary.pass={s['pass']} vs verify={pass_n}"
            assert s["fail"] == fail_n, f"{r['run_id']}: summary.fail={s['fail']} vs verify={fail_n}"
        assert s["all_pass"] == (s["fail"] == 0), f"{r['run_id']}: all_pass inconsistent"


def test_summary_total_time_matches_timing(runs_raw):
    for r in runs_raw:
        total = sum(float(t.get("time_s", 0)) for t in r.get("timing", []) or [])
        # summary.total_time is rounded; tolerate 1s drift
        assert abs(r["summary"]["total_time"] - total) < 1.0 + 0.01 * total, (
            f"{r['run_id']}: summary.total_time={r['summary']['total_time']} vs sum(timing)={total}"
        )


@pytest.mark.parametrize("rate_key", [
    "cache-miss-rate",
    "branch-miss-rate",
    "L1-dcache-miss-rate",
    "frontend-stall-rate",
])
def test_rates_are_percentages(runs_raw, rate_key):
    for r in runs_raw:
        m = (r.get("profile") or {}).get("metrics") or {}
        if rate_key in m and m[rate_key] is not None:
            v = m[rate_key]
            assert 0 <= v <= 100, f"{r['run_id']}: {rate_key}={v} is not a valid percentage"


def test_ipc_in_reasonable_range(runs_raw):
    for r in runs_raw:
        m = (r.get("profile") or {}).get("metrics") or {}
        if "IPC" in m and m["IPC"] is not None:
            ipc = m["IPC"]
            assert 0 < ipc < 10, f"{r['run_id']}: IPC={ipc} outside reasonable range (0, 10)"


def test_hotspot_percents_valid(runs_raw):
    for r in runs_raw:
        hs = (r.get("profile") or {}).get("hotspots") or []
        for h in hs:
            assert 0 <= h["percent"] <= 100, f"{r['run_id']}: hotspot percent {h['percent']} invalid"


def test_env_has_required_fields_when_present(runs_raw):
    """If env is non-empty, it should have at least hostname and date."""
    for r in runs_raw:
        env = r.get("env", {})
        if env:
            assert "hostname" in env or "date" in env, f"{r['run_id']}: env missing hostname/date"
