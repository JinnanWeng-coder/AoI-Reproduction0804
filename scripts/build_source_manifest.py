"""Build the immutable source provenance manifest for the reproduction."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


TARGET_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TARGET_ROOT.parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "src" / "AoI-V2X-IEEE-TVT-2023-reimplement" / "1-Modified_MADDPG_with_TDec"
REQUESTED_SOURCE_ROOT = WORKSPACE_ROOT / "src" / "AoI-V2X-IEEE-TVT-2023-main" / "1-Modified_MADDPG_with_TDec"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(SOURCE_ROOT.parent), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build() -> dict:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(SOURCE_ROOT)
    files = []
    for path in sorted(p for p in SOURCE_ROOT.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel = path.relative_to(SOURCE_ROOT).as_posix()
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_source_root": str(REQUESTED_SOURCE_ROOT),
        "source_root": str(SOURCE_ROOT),
        "path_mapping_reason": "The requested -main path was replaced by the clean Git clone -reimplement path.",
        "requested_path_exists": REQUESTED_SOURCE_ROOT.is_dir(),
        "source_git_commit": git_value("rev-parse", "HEAD"),
        "source_git_status": git_value("status", "--short", "--branch"),
        "file_count": len(files),
        "files": files,
    }


if __name__ == "__main__":
    output = TARGET_ROOT / "SOURCE_MANIFEST.json"
    output.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
