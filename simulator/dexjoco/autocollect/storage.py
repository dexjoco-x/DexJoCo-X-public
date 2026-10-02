"""Crash-resistant low-dimensional output storage for AutoCollect runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import zarr

from dexjoco.data.episode_store import ZarrEpisodeStore

from .core import LowDimEpisodeRecorder, SkillResult, _json_safe
from .verifiers import VerificationResult


def save_episode(
    *,
    output_dir: str | Path,
    task_id: str,
    seed: int,
    recorder: LowDimEpisodeRecorder,
    skill_result: SkillResult,
    verification: VerificationResult,
    source_demo: str | Path,
    recovery_result: SkillResult | None = None,
    backend: str = "recorded_native_action",
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save one episode as ``replay.zarr`` plus ``summary.json``."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if recorder.steps <= 0:
        raise ValueError("cannot save an episode with no transitions")

    status = "success" if skill_result.success and verification.success else "failure"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    final_dir = output_dir / f"episode_{timestamp}_{status}"
    temporary_dir = Path(tempfile.mkdtemp(prefix=".episode_tmp_", dir=output_dir))

    zarr_path = temporary_dir / "replay.zarr"
    store = ZarrEpisodeStore.create_empty(
        storage=zarr.DirectoryStore(str(zarr_path))
    )
    store.append_episode(recorder.episode_data(), compressors="disk")
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "task_id": task_id,
        "seed": int(seed),
        "control_hz": recorder.control_hz,
        "backend": str(backend),
        "vla_used": False,
        "source_demo": str(Path(source_demo).expanduser().resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        overlap = set(metadata).intersection(extra_metadata)
        if overlap:
            raise ValueError(
                f"extra_metadata cannot replace reserved keys: {sorted(overlap)}"
            )
        metadata.update(_json_safe(extra_metadata))
    store.root.attrs["dexjoco_autocollect_metadata"] = metadata

    summary = {
        "status": status,
        **metadata,
        "episode_dir": str(final_dir),
        "steps_saved": recorder.steps,
        "skill": skill_result.to_dict(),
        "verification": verification.to_dict(),
        "recovery": recovery_result.to_dict() if recovery_result else metadata.get("recovery"),
    }
    summary_path = temporary_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_dir, final_dir)
    return final_dir
