#!/usr/bin/env python3
"""
Build the Gadi-IQ / Setonix-IQ dashboard.

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
    built_at = datetime.now(timezone.utc).isoformat()
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    bi.write_text(json.dumps({
        "built_at": built_at,
        "version": version,
        "source": "tools/build.py",
    }, indent=2))

    # 4. Cache-bust: inject ?v=<version> onto the main.js module entry + CSS
    #    AND rewrite all relative ES-module import paths inside docs/js/ so
    #    every module URL is unique per build. GitHub Pages serves with a
    #    10-minute max-age cache; without busting, users see stale code.
    index_html = DOCS / "index.html"
    if index_html.exists():
        html = index_html.read_text()
        html = html.replace('js/main.js"', f'js/main.js?v={version}"')
        import re as _re
        html = _re.sub(r'href="(css/[^"?]+\.css)"', rf'href="\1?v={version}"', html)
        html = html.replace('<head>', f'<head>\n  <meta name="site-version" content="{version}">', 1)
        index_html.write_text(html)

    # Rewrite relative imports in every JS module under docs/js/
    import re as _re2
    import_re = _re2.compile(r"""((?:import|from)\s+(?:[^'"]*?\s+from\s+)?['"])((?:\./|\.\./)[^'"?]+\.js)(['"])""")
    dyn_import_re = _re2.compile(r"""(import\(['"])((?:\./|\.\./)[^'"?]+\.js)(['"]\))""")
    for js in (DOCS / "js").rglob("*.js"):
        text = js.read_text()
        new = import_re.sub(lambda m: f"{m.group(1)}{m.group(2)}?v={version}{m.group(3)}", text)
        new = dyn_import_re.sub(lambda m: f"{m.group(1)}{m.group(2)}?v={version}{m.group(3)}", new)
        if new != text:
            js.write_text(new)

    print(f"[build] OK → {DOCS} (v={version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
