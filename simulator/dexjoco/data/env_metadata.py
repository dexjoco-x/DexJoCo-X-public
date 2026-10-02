"""Metadata helpers for recorded DexJoCo demos."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except Exception:
        return ""


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _source_file(obj: Any) -> Path | None:
    try:
        source = inspect.getsourcefile(obj)
    except Exception:
        source = None
    return Path(source).resolve() if source else None


def _jsonable_flags(flags_obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in (
        "exp_name",
        "successes_needed",
        "out_dir",
        "env_hz",
        "render_mode",
        "randomize",
        "collection_randomization",
        "episode_time_limit",
        "episode_max_steps",
        "success_settle_steps",
        "camera_screen_effect",
        "ipad_screen_effect",
        "save_depth",
        "galr_device",
        "galr_max_projection_ms",
    ):
        try:
            value = getattr(flags_obj, name)
        except Exception:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        else:
            out[name] = str(value)
    return out


def _git_metadata(root: Path) -> dict[str, Any]:
    diff = _run_git(["diff", "--binary", "HEAD"], root)
    staged = _run_git(["diff", "--cached", "--binary", "HEAD"], root)
    status = _run_git(["status", "--short"], root)
    untracked = "\n".join(
        line[3:] for line in status.splitlines() if line.startswith("?? ")
    )
    dirty_payload = "\n".join((diff, staged, status))
    return {
        "commit": _run_git(["rev-parse", "HEAD"], root),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "status_short": status,
        "is_dirty": bool(status),
        "diff_sha256": _sha256_bytes(dirty_payload.encode("utf-8")),
        "untracked_paths_sha256": _sha256_bytes(untracked.encode("utf-8")),
    }


def build_collection_metadata(
    *,
    task_id: str,
    config: Any,
    env: Any,
    flags_obj: Any,
    collection_hand: str,
    collection_arm_layout: str,
    collection_randomization: str,
) -> dict[str, Any]:
    """Build JSON-compatible metadata for a newly recorded demo."""

    root = _repo_root()
    raw_env = getattr(env, "unwrapped", env)
    config_file = _source_file(config.__class__)
    env_file = _source_file(raw_env.__class__)
    xml_path = getattr(raw_env, "_xml_path", None)
    xml_file = Path(xml_path).resolve() if xml_path else None

    package = Path(__file__).resolve().parents[1]
    candidate_files = [
        Path(__file__).resolve(),
        package / "sim" / "mujoco_gym_env.py",
        package / "tasks" / "state_restorers.py",
        package / "tasks" / "native_delta_teleop.py",
        package / "sim" / "envs" / "panda_pick_bucket_native_delta_env.py",
        config_file,
        env_file,
        xml_file,
    ]
    file_hashes: dict[str, str] = {}
    for path in candidate_files:
        if path is None:
            continue
        path = Path(path)
        if path.exists() and path.is_file():
            file_hashes[_relpath(path, root)] = _file_sha256(path)

    return {
        "schema_version": 1,
        "task_id": task_id,
        "action_semantics": str(
            getattr(config, "action_semantics", "absolute_tcp_target")
        ),
        "rotation_semantics": str(
            getattr(config, "rotation_semantics", "absolute_quaternion_wxyz")
        ),
        "action_layout": list(getattr(config, "action_layout", ())),
        "arm_action_dim": int(getattr(config, "arm_action_dim", 7)),
        "hand_action_dim": int(getattr(config, "hand_action_dim", 0)),
        "delta_input_mode": str(getattr(config, "delta_input_mode", "")),
        "delta_clipping_mode": str(getattr(config, "delta_clipping_mode", "")),
        "accumulate_clipped_residual": bool(
            getattr(config, "accumulate_clipped_residual", False)
        ),
        "delta_position_limit": float(
            getattr(getattr(config, "teleop", None), "delta_position_limit", 0.0)
        ),
        "delta_rotation_limit": float(
            getattr(getattr(config, "teleop", None), "delta_rotation_limit", 0.0)
        ),
        "control_dt": float(getattr(raw_env, "control_dt", 0.0)),
        "physics_dt": float(getattr(raw_env, "physics_dt", 0.0)),
        "control_substeps": int(getattr(raw_env, "_n_substeps", 0)),
        "arm_pos_gain_base": float(getattr(raw_env, "_arm_pos_gain_base", 0.0)),
        "arm_ori_gain_base": float(getattr(raw_env, "_arm_ori_gain_base", 0.0)),
        "arm_damping_ratio": float(getattr(raw_env, "_arm_damping_ratio", 0.0)),
        "impact_velocity_timebase": str(
            getattr(raw_env, "_impact_velocity_timebase", "")
        ),
        "collection_hand": collection_hand,
        "collection_arm_layout": collection_arm_layout,
        "collection_randomization": collection_randomization,
        "config_class": f"{config.__class__.__module__}.{config.__class__.__name__}",
        "env_class": f"{raw_env.__class__.__module__}.{raw_env.__class__.__name__}",
        "xml_path": _relpath(xml_file, root) if xml_file else "",
        "flags": _jsonable_flags(flags_obj),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_metadata(root),
        "file_sha256": file_hashes,
    }


def attach_demo_metadata(root: Any, metadata: dict[str, Any]) -> None:
    """Attach metadata to a zarr root group using attrs only."""

    root.attrs["dexjoco_collection_metadata"] = metadata
    root.attrs["dexjoco_collection_metadata_json"] = json.dumps(
        metadata,
        ensure_ascii=True,
        sort_keys=True,
    )
    meta = root.require_group("meta")
    meta.attrs["dexjoco_collection_metadata"] = metadata
