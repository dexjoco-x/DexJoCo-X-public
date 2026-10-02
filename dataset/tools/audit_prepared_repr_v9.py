#!/usr/bin/env python3
"""Audit the public v9 FAAS/DexLatent Being-H training views."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


EXPECTED_CONTRACT_SHA256 = "57ec9c7b73d57e866b816924f3bea70b93a058ab379bd084bb8a1bb4526cc51c"
EXPECTED_DEXLATENT_SHA256 = "73b1c80497a493835e58ace3e3abe509906d63916a4c263f5191d585a38bbd7b"


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("release", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
release = args.release.resolve()
prepared = release / "prepared"

source_rows = []
source_hashes = {}
for path in sorted((release / "bundle/manifests").glob("*.json")):
    source_rows.extend(read(path))
    source_hashes[path.name] = digest(path)
expected_keys = {(row["task"], row["hand"], row["split"]) for row in source_rows}
expected_frames = sum(int(row["steps"]) for row in source_rows)
assert len(source_rows) == 2100
assert len(expected_keys) == 42
assert {row["split"] for row in source_rows} == {"train"}

reports = {}
for representation in ("faas", "dexlatent"):
    manifest = read(prepared / f"beingh05_{representation}_datasets.json")
    conversion = read(prepared / f"beingh05_{representation}_conversion_check.json")
    stats = read(prepared / f"beingh05_{representation}_train_stats.json")
    recipe = yaml.safe_load((prepared / f"beingh05_{representation}_train.yaml").read_text())

    assert conversion["passed"] is True
    assert conversion["episodes"] == 2100
    assert conversion["train_episodes"] == 2100
    assert conversion["test_split_used_for_statistics"] is False
    assert conversion["two_arm_encoding"] is True
    assert stats["representation"] == representation
    assert stats["codec_contract_sha256"] == EXPECTED_CONTRACT_SHA256
    assert stats["codec_checkpoint_sha256"] == (
        EXPECTED_DEXLATENT_SHA256 if representation == "dexlatent" else None
    )
    assert len(manifest) == len(expected_keys) == len(recipe) == 42
    assert {(row["task"], row["hand"], row["split"]) for row in manifest} == expected_keys
    assert all(int(row["episodes"]) == 50 for row in manifest)

    total_rows = total_episodes = 0
    min_value, max_value = float("inf"), float("-inf")
    nonfinite = masked_nonzero = 0
    stats_hashes = set()
    group_frames = {}
    for entry in manifest:
        group = release / entry["path"]
        info = read(group / "meta/info.json")
        stats_hashes.add(digest(group / "meta/raw_train_statistics.json"))
        files = sorted(group.glob("data/chunk-*/*.parquet"))
        assert len(files) == int(entry["episodes"]) == int(info["total_episodes"]) == 50
        rows = 0
        for path in files:
            table = pq.read_table(
                path,
                columns=["observation.state", "action", "observation.state_mask", "action.valid_mask"],
            )
            q = np.asarray(table["observation.state"].to_pylist(), np.float32)
            u = np.asarray(table["action"].to_pylist(), np.float32)
            qm = np.asarray(table["observation.state_mask"].to_pylist(), bool)
            um = np.asarray(table["action.valid_mask"].to_pylist(), bool)
            assert q.shape == qm.shape == u.shape == um.shape
            assert q.ndim == 2 and q.shape[1] == 200
            nonfinite += int((~np.isfinite(q)).sum() + (~np.isfinite(u)).sum())
            masked_nonzero += int(np.count_nonzero(q[~qm]) + np.count_nonzero(u[~um]))
            min_value = min(min_value, float(q.min()), float(u.min()))
            max_value = max(max_value, float(q.max()), float(u.max()))
            rows += len(q)
        assert rows == int(entry["frames"]) == int(info["total_frames"])
        group_frames[(entry["task"], entry["hand"])] = rows
        total_rows += rows
        total_episodes += len(files)

    for name, config in recipe.items():
        matches = [
            key for key in group_frames
            if name == f"benchmark7_{key[0]}_{key[1]}"
        ]
        assert len(matches) == 1
        assert config["sampling_strategy"] == "step"
        assert int(config["weight"]) == group_frames[matches[0]]

    passed = (
        total_episodes == 2100
        and total_rows == expected_frames
        and nonfinite == 0
        and masked_nonzero == 0
        and min_value >= -1.000001
        and max_value <= 1.000001
        and len(stats_hashes) == 1
    )
    reports[representation] = {
        "passed": passed,
        "groups": len(manifest),
        "episodes": total_episodes,
        "frames": total_rows,
        "all_train": True,
        "episodes_per_group": 50,
        "array_width": 200,
        "finite": nonfinite == 0,
        "masked_values_are_zero": masked_nonzero == 0,
        "min": min_value,
        "max": max_value,
        "shared_train_stats_hash_count": len(stats_hashes),
        "codec_contract_sha256": stats["codec_contract_sha256"],
        "codec_checkpoint_sha256": stats["codec_checkpoint_sha256"],
        "source_manifest_sha256": source_hashes,
        "sampling": "frame-proportional step sampling",
    }
    if not passed:
        raise SystemExit(f"{representation} audit failed")

result = {
    "passed": all(report["passed"] for report in reports.values()),
    "source": {"groups": 42, "episodes": 2100, "frames": expected_frames, "split": "train"},
    "representations": reports,
}
output = args.output or (release / "PREPARED_REPRESENTATION_VERIFICATION.json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
