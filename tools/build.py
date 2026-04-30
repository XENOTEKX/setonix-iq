#!/usr/bin/env python3
"""
Build the Gadi-IQ / Setonix-IQ dashboard.

Steps:
  1. Normalize logs/ → web/data/ (runs, profiles, indexes, manifest).
  2. Smart-sync web/ → docs/ (only copy changed files).
  3. Split heavy per-run blobs (folded_stacks, memory_timeseries) into a
     companion docs/data/runs/<id>.profile.json and minify the main run JSON.
  4. Stamp build-info JSON and cache-bust JS/CSS imports in index.html.
  5. Rewrite relative ES-module import paths inside docs/js/ with ?v=<stamp>.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
WEB = ROOT / "web"
DOCS = ROOT / "docs"

# Keys stripped from the main run JSON and saved to <id>.profile.json instead.
# Only needed by the Profiling/Flamegraph pages — lazy-loaded on demand.
HEAVY_BLOB_KEYS = {"folded_stacks", "memory_timeseries"}


def run(cmd: list[str]) -> None:
    print("[build] $", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        sys.exit(r.returncode)


# ── Smart incremental copy ─────────────────────────────────────────────────────

def _needs_copy(src: Path, dst: Path) -> bool:
    """Return True if src should be copied over dst.

    A destination that is newer than its source (e.g. post-processed by
    cache-busting) is considered up-to-date and does NOT need re-copying.
    """
    if not dst.exists():
        return True
    return src.stat().st_mtime > dst.stat().st_mtime + 0.01


def sync_tree(src: Path, dst: Path, exclude: set = None) -> tuple[int, int]:
    """Mirror src → dst; only copy files that changed (mtime or size).

    Removes dst-only files so deletions propagate.
    ``exclude`` is an optional set of directory names to skip at the top level.
    Returns (copied, skipped).
    """
    if exclude is None:
        exclude = set()
    dst.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    src_names = {p.name for p in src.iterdir()}

    # Prune stale dst entries (except excluded dirs managed elsewhere)
    for item in list(dst.iterdir()):
        if item.name in exclude:
            continue
        if item.name not in src_names:
            shutil.rmtree(item) if item.is_dir() else item.unlink()

    for src_item in src.iterdir():
        if src_item.name in exclude:
            continue
        dst_item = dst / src_item.name
        if src_item.is_dir():
            c, s = sync_tree(src_item, dst_item)
            copied += c
            skipped += s
        elif _needs_copy(src_item, dst_item):
            shutil.copy2(src_item, dst_item)
            copied += 1
        else:
            skipped += 1

    return copied, skipped


# ── Heavy-blob split + JSON minification ──────────────────────────────────────

def split_run_blobs(docs_runs: Path, web_runs: Path) -> tuple[int, int]:
    """Sync web/data/runs/ → docs/data/runs/ with blob splitting and minification.

    For each run JSON in web_runs:
      - Copy only if source changed (size/mtime check against the blob-companion
        sentinel <id>.profile.json.stamp or the companion file itself).
      - Strip HEAVY_BLOB_KEYS into <id>.profile.json.
      - Write compact main JSON.

    Returns (split_count, bytes_saved_by_minification).
    """
    docs_runs.mkdir(parents=True, exist_ok=True)
    split_count = bytes_saved = 0

    src_names = {p.name for p in web_runs.glob("*.json")}

    # Remove stale docs files (run was deleted from logs)
    for dst_item in list(docs_runs.glob("*.json")):
        base = dst_item.name
        if base.endswith(".profile.json"):
            stem = base[: -len(".profile.json")]
            if f"{stem}.json" not in src_names:
                dst_item.unlink()
        elif base not in src_names:
            dst_item.unlink()
            companion = docs_runs / (dst_item.stem + ".profile.json")
            if companion.exists():
                companion.unlink()

    for src_file in sorted(web_runs.glob("*.json")):
        dst_file = docs_runs / src_file.name
        companion = docs_runs / (src_file.stem + ".profile.json")

        # Use the companion mtime as the "already processed" sentinel:
        # if companion exists and src hasn't been modified since companion was
        # last written, both main + companion are already up to date → skip.
        if companion.exists() and dst_file.exists():
            src_mtime = src_file.stat().st_mtime
            cmp_mtime = companion.stat().st_mtime
            if src_mtime <= cmp_mtime + 0.01:
                continue

        try:
            data = json.loads(src_file.read_bytes())
        except (json.JSONDecodeError, OSError):
            continue

        profile = data.get("profile")
        blobs = {}
        if isinstance(profile, dict):
            for k in list(profile.keys()):
                if k in HEAVY_BLOB_KEYS:
                    blobs[k] = profile.pop(k)

        if blobs:
            companion.write_text(json.dumps(blobs, separators=(",", ":")))
            # Touch companion to match source mtime so next build skips it
            import os as _os
            src_mtime = src_file.stat().st_mtime
            _os.utime(companion, (src_mtime, src_mtime))
            split_count += 1
        elif companion.exists():
            companion.unlink()

        orig_size = src_file.stat().st_size
        compact = json.dumps(data, separators=(",", ":"))
        dst_file.write_text(compact)
        bytes_saved += orig_size - len(compact.encode())

    return split_count, bytes_saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    if not WEB.exists():
        print(f"[build] FATAL: {WEB} does not exist", file=sys.stderr)
        return 2

    # 1. Normalize data
    run([sys.executable, str(TOOLS / "normalize.py")])

    # 2. Smart-sync web/ → docs/ (exclude data/ — handled separately below)
    print(f"[build] syncing {WEB} → {DOCS}")
    copied, skipped = sync_tree(WEB, DOCS, exclude={"data"})
    print(f"[build] sync: {copied} copied, {skipped} unchanged")

    # 3. Stamp build info
    built_at = datetime.now(timezone.utc).isoformat()
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    (DOCS / "build-info.json").write_text(json.dumps({
        "built_at": built_at,
        "version": version,
        "source": "tools/build.py",
    }, indent=2))

    # 4. Sync data/ with blob splitting + minification for per-run files.
    #    Non-run data files (indexes, profiles, manifest) are copied normally.
    web_data = WEB / "data"
    docs_data = DOCS / "data"
    if web_data.is_dir():
        # Copy everything except runs/ via normal sync
        sync_tree(web_data, docs_data, exclude={"runs"})
        # Runs: blob-split + minify
        web_runs = web_data / "runs"
        docs_runs = docs_data / "runs"
        if web_runs.is_dir():
            split, saved = split_run_blobs(docs_runs, web_runs)
            if split > 0:
                print(f"[build] data: {split} runs split, {max(0, saved) // 1024} KB saved")

    # 5. Cache-bust index.html (main.js entry + CSS links)
    index_html = DOCS / "index.html"
    if index_html.exists():
        html = index_html.read_text()
        html = html.replace('js/main.js"', f'js/main.js?v={version}"')
        html = re.sub(r'href="(css/[^"?]+\.css)"', rf'href="\1?v={version}"', html)
        html = html.replace('<head>', f'<head>\n  <meta name="site-version" content="{version}">', 1)
        index_html.write_text(html)

    # 6. Rewrite relative ES-module imports inside docs/js/ with ?v=<stamp>
    import_re = re.compile(
        r"""((?:import|from)\s+(?:[^'"]*?\s+from\s+)?['"])((?:\./|\.\./)[^'"?]+\.js)(['"])"""
    )
    dyn_import_re = re.compile(r"""(import\(['"])((?:\./|\.\./)[^'"?]+\.js)(['"]\))""")
    bust_count = 0
    for js in (DOCS / "js").rglob("*.js"):
        text = js.read_text()
        new = import_re.sub(lambda m: f"{m.group(1)}{m.group(2)}?v={version}{m.group(3)}", text)
        new = dyn_import_re.sub(
            lambda m: f"{m.group(1)}{m.group(2)}?v={version}{m.group(3)}", new
        )
        if new != text:
            js.write_text(new)
            bust_count += 1
    print(f"[build] cache-bust: {bust_count} JS files updated")

    print(f"[build] OK → {DOCS} (v={version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

