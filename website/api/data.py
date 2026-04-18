#!/usr/bin/env python3
"""
Simple JSON API that reads pipeline results and profiles from symlinked dirs.
Called by the dashboard frontend via fetch().
"""
import json
import os
import re
import glob
from pathlib import Path

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
PROFILES_DIR = os.path.join(os.path.dirname(__file__), '..', 'profiles')
DEEP_PROFILES_DIR = os.path.join(os.path.dirname(__file__), '..', 'deep_profiles')
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')


def slurm_id_to_datetime(slurm_id, env_info):
    """Convert a SLURM job ID to a date-time string using the env date field.
    Returns format: 2026-04-18_201515 (sortable, filesystem-safe)."""
    date_str = env_info.get('date', '')
    if date_str:
        # Parse ISO format: 2026-04-18T20:15:15+08:00
        m = re.match(r'(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})', date_str)
        if m:
            return '%s_%s%s%s' % (m.group(1), m.group(2), m.group(3), m.group(4))
    # Fallback: use slurm_id prefixed with "run_"
    return 'run_%s' % slurm_id


def parse_time_log(filepath):
    rows = []
    with open(filepath) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cmd = parts[0].strip()
                short = re.sub(r'.*/iqtree3\s+', 'iqtree3 ', cmd)
                short = re.sub(r'test_scripts/test_data/', '', short)
                rows.append({
                    'command': short[:80],
                    'time_s': float(parts[1]),
                    'memory_kb': float(parts[2]) if len(parts) > 2 else 0,
                })
    return rows


def parse_verify_log(filepath):
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('PASS:'):
                m = re.match(r'PASS:\s+(\S+)\s+--\s+Expected:\s+([\-\d.]+),\s+Reported:\s+([\-\d.]+),\s+Abs-diff:\s+([\d.]+)', line)
                if m:
                    results.append({
                        'status': 'pass',
                        'file': m.group(1).replace('test_scripts/test_data/', ''),
                        'expected': float(m.group(2)),
                        'reported': float(m.group(3)),
                        'diff': float(m.group(4)),
                    })
            elif line.startswith('FAIL:'):
                m = re.match(r'FAIL:\s+(\S+)\s+--\s+Expected:\s+([\-\d.]+),\s+Reported:\s+([\-\d.]+),\s+Abs-diff:\s+([\d.]+)', line)
                if m:
                    results.append({
                        'status': 'fail',
                        'file': m.group(1).replace('test_scripts/test_data/', ''),
                        'expected': float(m.group(2)),
                        'reported': float(m.group(3)),
                        'diff': float(m.group(4)),
                    })
    return results


def parse_env_info(filepath):
    info = {}
    with open(filepath) as f:
        for line in f:
            if ':' in line:
                key, _, val = line.partition(':')
                info[key.strip()] = val.strip()
    return info


def get_all_runs():
    """Return structured data for all pipeline runs.

    Reads raw log files from website/results and website/profiles (Setonix
    symlinks).  When those directories are unavailable (local dev), falls
    back to individual per-run JSON files in logs/runs/<run_id>.json.
    """
    # If raw results dir doesn't exist, fall back to cached per-run logs
    if not os.path.isdir(RESULTS_DIR):
        runs_dir = os.path.join(LOGS_DIR, 'runs')
        if os.path.isdir(runs_dir):
            runs = []
            for f in sorted(glob.glob(os.path.join(runs_dir, '*.json'))):
                with open(f) as fh:
                    runs.append(json.load(fh))
            return runs
        return []

    runs = []
    for tlog in sorted(glob.glob(os.path.join(RESULTS_DIR, 'time_log_*.tsv'))):
        slurm_id = Path(tlog).stem.replace('time_log_', '')
        timing = parse_time_log(tlog)

        verify_file = os.path.join(RESULTS_DIR, f'verify_{slurm_id}.txt')
        verify = parse_verify_log(verify_file) if os.path.exists(verify_file) else []

        env_file = os.path.join(RESULTS_DIR, f'env_{slurm_id}.txt')
        env_info = parse_env_info(env_file) if os.path.exists(env_file) else {}

        gpu_file = os.path.join(RESULTS_DIR, f'gpu_info_{slurm_id}.txt')
        gpu_info = ''
        if os.path.exists(gpu_file):
            with open(gpu_file) as f:
                gpu_info = f.read()

        profile_file = os.path.join(PROFILES_DIR, f'perf_stat_{slurm_id}.json')
        profile = {}
        if os.path.exists(profile_file):
            with open(profile_file) as f:
                profile = json.load(f)

        pass_count = sum(1 for v in verify if v['status'] == 'pass')
        fail_count = sum(1 for v in verify if v['status'] == 'fail')
        total_time = sum(r['time_s'] for r in timing)

        run_id = slurm_id_to_datetime(slurm_id, env_info)

        runs.append({
            'run_id': run_id,
            'slurm_id': slurm_id,
            'timing': timing,
            'verify': verify,
            'env': env_info,
            'gpu_info': gpu_info,
            'profile': profile,
            'summary': {
                'pass': pass_count,
                'fail': fail_count,
                'total_time': round(total_time, 2),
                'all_pass': fail_count == 0,
            }
        })
    return runs


def profile_id_to_datetime(slurm_id, sys_info):
    """Convert a SLURM job ID to date-time string using system info date."""
    date_str = sys_info.get('date', '')
    if date_str:
        m = re.match(r'(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})', date_str)
        if m:
            return '%s_%s%s%s' % (m.group(1), m.group(2), m.group(3), m.group(4))
    return 'profile_%s' % slurm_id


def get_all_profiles():
    """Return structured data for all deep profiling sessions.

    Reads deep_profile_*.json from website/deep_profiles/ (Setonix symlink).
    Falls back to logs/profiles/<id>.json for local/offline use.
    """
    profiles = []

    # Try reading from symlinked deep_profiles directory (Setonix)
    if os.path.isdir(DEEP_PROFILES_DIR):
        for f in sorted(glob.glob(os.path.join(DEEP_PROFILES_DIR, 'deep_profile_*.json'))):
            try:
                with open(f) as fh:
                    p = json.load(fh)
                # Convert profile_id from SLURM ID to date-time
                raw_id = p.get('profile_id', '')
                sys_info = p.get('system', {})
                p['profile_id'] = profile_id_to_datetime(raw_id, sys_info)
                p['slurm_id'] = raw_id
                profiles.append(p)
            except (json.JSONDecodeError, IOError):
                continue

    # Fallback: read from committed logs/profiles/
    if not profiles:
        profiles_dir = os.path.join(LOGS_DIR, 'profiles')
        if os.path.isdir(profiles_dir):
            for f in sorted(glob.glob(os.path.join(profiles_dir, '*.json'))):
                try:
                    with open(f) as fh:
                        profiles.append(json.load(fh))
                except (json.JSONDecodeError, IOError):
                    continue

    return profiles


if __name__ == '__main__':
    print('=== Runs ===')
    print(json.dumps(get_all_runs(), indent=2))
    print('\n=== Profiles ===')
    print(json.dumps(get_all_profiles(), indent=2))
