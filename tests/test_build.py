"""
End-to-end build test: normalize + build should produce a usable docs/ dir.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_normalize_produces_index(tmp_path, monkeypatch):
    # Run normalize.py in-tree (writes to web/data/)
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "normalize.py")],
        check=False, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"normalize.py failed:\n{r.stdout}\n{r.stderr}"
    idx = ROOT / "web" / "data" / "runs.index.json"
    assert idx.exists(), "runs.index.json not generated"
    data = json.loads(idx.read_text())
    assert isinstance(data, list), "runs.index.json must be a list"
    assert len(data) >= 1, "runs.index.json is empty"
    # Every entry must have the fields the frontend relies on
    for r in data:
        assert "run_id" in r
        assert "wall_s" in r


def test_manifest_has_version():
    manifest = ROOT / "web" / "data" / "manifest.json"
    if not manifest.exists():
        # Run normalize if manifest missing (CI runs tests after normalize)
        subprocess.run([sys.executable, str(ROOT / "tools" / "normalize.py")], check=True)
    m = json.loads(manifest.read_text())
    assert "schema_version" in m
    assert "generated_at" in m
    assert m["runs"] >= 0
