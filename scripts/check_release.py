"""Fail when the repository contains private or non-source release artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensoralpha.release_policy import scan_repository


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    findings = scan_repository(args.root)
    for finding in findings:
        location = f":{finding.line}" if finding.line is not None else ""
        print(f"{finding.kind}: {finding.path}{location}")
    if findings:
        print(f"Release policy failed with {len(findings)} finding(s).")
        return 1
    print("Release policy passed: no private or forbidden artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
