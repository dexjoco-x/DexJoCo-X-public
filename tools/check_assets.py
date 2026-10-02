#!/usr/bin/env python3
"""Verify audited asset directory trees against docs/ASSET_INVENTORY.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_record(path: Path) -> tuple[int, int, str]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(file_sha256(item).encode())
        digest.update(b"\n")
    return len(files), sum(item.stat().st_size for item in files), digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.inventory is None:
        from runtime_assets import check
        result = check(ROOT, json.loads((ROOT / 'assets/LOCAL_ASSET_MANIFEST.json').read_text()))
        print(json.dumps(result, indent=2))
        return int(not result['passed'])
    inventory = json.loads(args.inventory.read_text())
    failures = []
    for expected in inventory["assets"]:
        path = ROOT / expected["path"]
        if not path.is_dir():
            failures.append({"name": expected["name"], "error": "missing directory"})
            continue
        files, size, digest = tree_record(path)
        actual = {"files": files, "bytes": size, "tree_sha256": digest}
        wanted = {key: expected[key] for key in actual}
        if actual != wanted:
            failures.append({"name": expected["name"], "expected": wanted, "actual": actual})
    result = {
        "passed": not failures,
        "assets_checked": len(inventory["assets"]),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
