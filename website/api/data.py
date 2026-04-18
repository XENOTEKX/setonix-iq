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
    """Return structured data for all pipeline runs."""
    runs = []
    for tlog in sorted(glob.glob(os.path.join(RESULTS_DIR, 'time_log_*.tsv'))):
        rid = Path(tlog).stem.replace('time_log_', '')
        timing = parse_time_log(tlog)

        verify_file = os.path.join(RESULTS_DIR, f'verify_{rid}.txt')
        verify = parse_verify_log(verify_file) if os.path.exists(verify_file) else []

        env_file = os.path.join(RESULTS_DIR, f'env_{rid}.txt')
        env_info = parse_env_info(env_file) if os.path.exists(env_file) else {}

        gpu_file = os.path.join(RESULTS_DIR, f'gpu_info_{rid}.txt')
        gpu_info = ''
        if os.path.exists(gpu_file):
            with open(gpu_file) as f:
                gpu_info = f.read()

        profile_file = os.path.join(PROFILES_DIR, f'perf_stat_{rid}.json')
        profile = {}
        if os.path.exists(profile_file):
            with open(profile_file) as f:
                profile = json.load(f)

        pass_count = sum(1 for v in verify if v['status'] == 'pass')
        fail_count = sum(1 for v in verify if v['status'] == 'fail')
        total_time = sum(r['time_s'] for r in timing)

        runs.append({
            'run_id': rid,
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


if __name__ == '__main__':
    print(json.dumps(get_all_runs(), indent=2))
