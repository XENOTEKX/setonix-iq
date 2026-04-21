#!/usr/bin/env python3
"""
Build the Setonix-IQ dashboard.

Steps:
  1. Normalize logs/ → web/data/ (runs, profiles, indexes, manifest).
  2. Copy web/ (static assets + JS modules + CSS) → docs/.
  3. Write a small build-info JSON into docs/data/manifest.json.

Replaces the legacy serve.py template-splicing approach.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
WEB = ROOT / "web"
DOCS = ROOT / "docs"


def run(cmd: list[str]) -> None:
    print("[build] $", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        sys.exit(r.returncode)


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    if not WEB.exists():
        print(f"[build] FATAL: {WEB} does not exist", file=sys.stderr)
        return 2

    # 1. Normalize data
    run([sys.executable, str(TOOLS / "normalize.py")])

    # 2. Mirror web/ to docs/
    print(f"[build] mirroring {WEB} → {DOCS}")
    copytree(WEB, DOCS)

    # 3. Stamp build info
    bi = DOCS / "build-info.json"
    bi.write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "tools/build.py",
    }, indent=2))

    print(f"[build] OK → {DOCS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
