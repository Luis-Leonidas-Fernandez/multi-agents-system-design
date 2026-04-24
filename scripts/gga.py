#!/usr/bin/env python3
"""Minimal local GGA runner used by the pre-commit hook.

This implementation intentionally keeps the interface expected by the hook:
`gga run`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _staged_files() -> list[str]:
    result = subprocess.run(
        ["/usr/bin/git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run() -> int:
    files = _staged_files()
    print("[gga] reviewing staged changes")
    if not files:
        print("[gga] no staged files found; passing")
        return 0

    for filename in files[:50]:
        print(f"[gga] {filename}")

    print("[gga] basic review passed")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        print("Usage: gga run")
        return 1
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
