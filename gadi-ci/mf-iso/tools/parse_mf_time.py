#!/usr/bin/env python3
"""parse_mf_time.py — parse MF-TIME / MF-MPI-DIAG markers from an MF-iso run.

Usage:
    parse_mf_time.py <work_dir>

Reads the work_dir's mf_time.log (per-model markers) and mf_diag.log (FCA
state + filterRatesMPI broadcast events) and prints a per-rank summary
plus a unified timeline.

The per-rank summary tells us:
  - how many models each rank evaluated
  - the mean / max per-model wall (a heavy +F or +R10 family balloons max)
  - when the rank reached the filterRatesMPI broadcast (key Phase 0.5 metric)
  - how many models it evaluated AFTER the broadcast (should be the
    surviving G4 variants, ~28 on this alignment)

The timeline tells us:
  - when rank N started its first model
  - when each rank reached the collective barrier
  - the rank-0-idle window before the broadcast (the Phase 0.5/0.6 bug
    that earlier work was chasing)
"""
import sys
import re
import os
from pathlib import Path
from collections import defaultdict


def parse_mf_time(path: Path):
    """Parse MF-TIME lines into list-of-dict per rank."""
    if not path.exists():
        return {}
    pat = re.compile(
        r'^MF-TIME:\s+rank\s+(\d+)\s+'
        r'model=(\d+)\s+'
        r'name=(\S+)\s+'
        r'subst=(\S+)\s+'
        r'rate=(\S+)\s+'
        r'start=([\d.]+)\s+'
        r'end=([\d.]+)\s+'
        r'dt=([\d.]+)\s+'
        r'score=(-?[\d.eE+-]+)\s+'
        r'ref_remaining=(-?\d+)'
    )
    per_rank = defaultdict(list)
    with open(path, errors="replace") as fp:
        for line in fp:
            m = pat.match(line.rstrip())
            if not m:
                continue
            rank, model, name, subst, rate, t0, t1, dt, score, ref_rem = m.groups()
            per_rank[int(rank)].append({
                "model": int(model),
                "name": name,
                "subst": subst,
                "rate": rate,
                "start": float(t0),
                "end": float(t1),
                "dt": float(dt),
                "score": float(score),
                "ref_remaining": int(ref_rem),
            })
    return dict(per_rank)


def parse_mf_diag(path: Path):
    """Parse MF-MPI-DIAG lines."""
    if not path.exists():
        return []
    events = []
    with open(path, errors="replace") as fp:
        for line in fp:
            line = line.rstrip()
            if "filterRatesMPI fired" in line:
                m = re.search(
                    r'rank\s+(\d+)/\d+\s+filterRatesMPI fired at model=(\d+)'
                    r'\s+ref_subst=(\S+)\s+\|bcast_ok_rates\|=(\d+)'
                    r'\s+local_pruned=(\d+)\s+best_score=(-?[\d.eE+-]+)',
                    line)
                if m:
                    events.append({
                        "type": "bcast_fire",
                        "rank": int(m.group(1)),
                        "model": int(m.group(2)),
                        "ref_subst": m.group(3),
                        "ok_rates_size": int(m.group(4)),
                        "local_pruned": int(m.group(5)),
                        "best_score": float(m.group(6)),
                    })
            elif "filterRatesMPI_enabled=" in line:
                events.append({"type": "gate", "raw": line})
            elif " owns " in line and "projected_cost=" in line:
                m = re.search(
                    r'rank\s+(\d+)/(\d+)\s+owns\s+(\d+)\s+groups,\s+(\d+)/(\d+)\s+models,'
                    r'\s+projected_cost=([\d.eE+-]+)\s+ref_subst=(\S+)'
                    r'\s+ref_remaining=(\d+)', line)
                if m:
                    events.append({
                        "type": "dispatch",
                        "rank": int(m.group(1)),
                        "nranks": int(m.group(2)),
                        "groups": int(m.group(3)),
                        "own_models": int(m.group(4)),
                        "total_models": int(m.group(5)),
                        "projected_cost": float(m.group(6)),
                        "ref_subst": m.group(7),
                        "ref_remaining": int(m.group(8)),
                    })
    return events


def summarise(per_rank, events):
    print(f"# MF-iso run summary — ranks observed: {sorted(per_rank.keys())}")
    print()

    # Find the broadcast event per rank.
    bcast_at = {e["rank"]: e for e in events if e.get("type") == "bcast_fire"}

    print("# Dispatch (MF-MPI-DIAG: rank N owns G groups, M/T models, ...):")
    for e in events:
        if e.get("type") == "dispatch":
            print(f"  rank {e['rank']}/{e['nranks']}: "
                  f"{e['groups']} groups, {e['own_models']}/{e['total_models']} models, "
                  f"proj_cost={e['projected_cost']:.3e}, ref_subst={e['ref_subst']}, "
                  f"ref_remaining={e['ref_remaining']}")
        elif e.get("type") == "gate":
            print(f"  gate: {e['raw']}")
    print()

    print("# Per-rank model evaluation summary:")
    print(f"  {'rank':>4}  {'#models':>8}  {'total_s':>10}  {'mean_s':>8}  "
          f"{'max_s':>8}  {'t0':>10}  {'t_bcast':>10}  {'#pre_bcast':>10}  {'#post_bcast':>11}")
    if not per_rank:
        print("  (no MF-TIME lines found)")
    for rank in sorted(per_rank.keys()):
        rows = per_rank[rank]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["start"])
        t0 = rows_sorted[0]["start"]
        total = sum(r["dt"] for r in rows_sorted)
        mean = total / len(rows_sorted)
        mx = max(r["dt"] for r in rows_sorted)
        # Broadcast model index for this rank.
        bcast_model = bcast_at.get(rank, {}).get("model")
        if bcast_model is not None:
            # The broadcast fires after the model finishes; bcast time is
            # this model's end.
            bcast_t = None
            pre = post = 0
            for r in rows_sorted:
                if r["model"] == bcast_model:
                    bcast_t = r["end"]
                if bcast_t is None:
                    pre += 1
                else:
                    post += 1
            bcast_rel = (bcast_t - t0) if bcast_t else None
            print(f"  {rank:>4}  {len(rows_sorted):>8}  "
                  f"{total:>10.2f}  {mean:>8.3f}  {mx:>8.3f}  "
                  f"{0.0:>10.2f}  {bcast_rel:>10.2f}  {pre:>10}  {post:>11}")
        else:
            print(f"  {rank:>4}  {len(rows_sorted):>8}  "
                  f"{total:>10.2f}  {mean:>8.3f}  {mx:>8.3f}  "
                  f"{0.0:>10.2f}  {'-':>10}  {'-':>10}  {'-':>11}")
    print()

    print("# Phase 0.5 broadcast events:")
    if not bcast_at:
        print("  (NO filterRatesMPI broadcasts observed -- gate disabled or "
              "MPI np=1?)")
    for rank in sorted(bcast_at.keys()):
        b = bcast_at[rank]
        print(f"  rank {b['rank']}: at model={b['model']}, "
              f"|ok_rates|={b['ok_rates_size']}, "
              f"local_pruned={b['local_pruned']}, "
              f"ref_subst={b['ref_subst']}, "
              f"best_score={b['best_score']:.3f}")
    print()

    # Convergence diagnostic — how far apart did the ranks reach the bcast?
    if len(bcast_at) >= 2:
        ranks = sorted(bcast_at.keys())
        # Need absolute times to compute spread.
        rank_bcast_t = {}
        for rank in ranks:
            rows = sorted(per_rank.get(rank, []), key=lambda r: r["start"])
            t0 = rows[0]["start"] if rows else None
            bcast_model = bcast_at[rank]["model"]
            for r in rows:
                if r["model"] == bcast_model:
                    rank_bcast_t[rank] = (r["end"], r["end"] - t0)
                    break
        if len(rank_bcast_t) == len(ranks):
            rel_times = [rank_bcast_t[r][1] for r in ranks]
            spread = max(rel_times) - min(rel_times)
            print(f"# Broadcast-arrival spread (Phase 0.6 metric): {spread:.2f} s")
            print(f"  (ideal: <30 s; Phase 0.5 baseline w/o ref-priority: ~370 s)")


def main():
    if len(sys.argv) != 2:
        print("usage: parse_mf_time.py <work_dir>", file=sys.stderr)
        sys.exit(2)
    work = Path(sys.argv[1])
    if not work.is_dir():
        print(f"not a directory: {work}", file=sys.stderr)
        sys.exit(2)

    mf_time_path = work / "mf_time.log"
    mf_diag_path = work / "mf_diag.log"

    per_rank = parse_mf_time(mf_time_path)
    events = parse_mf_diag(mf_diag_path)

    summarise(per_rank, events)


if __name__ == "__main__":
    main()
