#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    workspace = os.environ.get("YTS_WORKSPACE_DIR", "").strip()
    cwd = workspace or str(repo_root / "artifacts" / "yts_workspace")
    os.makedirs(cwd, exist_ok=True)

    yts_bin = shutil.which("yts")
    if not yts_bin:
        print("yts CLI not found in PATH", file=sys.stderr)
        return 127

    result = subprocess.run([yts_bin, *sys.argv[1:]], cwd=cwd)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
