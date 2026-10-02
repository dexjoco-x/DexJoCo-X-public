"""Append-only skill experience memory and bounded evolution proposals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

import numpy as np

from .core import _json_safe


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"append-only record already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(payload),
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link creation is atomic and cannot replace an existing record.
        os.link(temporary_path, path)
        temporary_path.unlink()
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class SkillExperience:
    """One compact, searchable outcome linked to its full trace evidence."""

    experience_id: str
    created_at_utc: str
    task_id: str
    skill_name: str
    skill_version: str
    success: bool
    parameters: Mapping[str, Any]
    scene_summary: Mapping[str, Any]
    skill_metrics: Mapping[str, Any]
    verification: Mapping[str, Any]
    trace_id: str
    trace_path: str
    experiment_id: str | None = None
    study_id: str | None = None
    failure_stage: str | None = None
    failure_codes: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    eligible_as_training_data: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for label, value in (
            ("experience_id", self.experience_id),
            ("created_at_utc", self.created_at_utc),
            ("task_id", self.task_id),
            ("skill_name", self.skill_name),
            ("skill_version", self.skill_version),
            ("trace_id", self.trace_id),
            ("trace_path", self.trace_path),
        ):
            if not str(value).strip():
                raise ValueError(f"skill experience {label} must not be empty")
        if self.success and (self.failure_stage is not None or self.failure_codes):
            raise ValueError("a successful experience cannot contain failure labels")
        if not self.success and not self.failure_stage:
            raise ValueError("a failed experience requires failure_stage")
        if self.eligible_as_training_data and not self.success:
            raise ValueError("a failed experience cannot be eligible training data")
        _canonical_json(self.to_dict())

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        skill_name: str,
        skill_version: str,
        success: bool,
        parameters: Mapping[str, Any],
        scene_summary: Mapping[str, Any],
        skill_metrics: Mapping[str, Any],
        verification: Mapping[str, Any],
        trace_id: str,
        trace_path: str | Path,
        experiment_id: str | None = None,
        study_id: str | None = None,
        failure_stage: str | None = None,
        failure_codes: Sequence[str] = (),
        artifact_refs: Sequence[str | Path] = (),
        eligible_as_training_data: bool = False,
        tags: Sequence[str] = (),
    ) -> "SkillExperience":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        return cls(
            experience_id=f"skill_exp_{timestamp}_{uuid.uuid4().hex[:10]}",
            created_at_utc=_utc_now(),
            task_id=str(task_id),
            skill_name=str(skill_name),
            skill_version=str(skill_version),
            success=bool(success),
            parameters=dict(parameters),
            scene_summary=dict(scene_summary),
            skill_metrics=dict(skill_metrics),
            verification=dict(verification),
            trace_id=str(trace_id),
            trace_path=str(Path(trace_path).expanduser().resolve()),
            experiment_id=experiment_id,
            study_id=study_id,
            failure_stage=failure_stage,
            failure_codes=tuple(str(code) for code in failure_codes if code),
            artifact_refs=tuple(
                str(Path(path).expanduser().resolve()) for path in artifact_refs
            ),
            eligible_as_training_data=bool(eligible_as_training_data),
            tags=tuple(str(tag) for tag in tags),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experience_id": self.experience_id,
            "created_at_utc": self.created_at_utc,
            "task_id": self.task_id,
            "skill_name": self.skill_name,
            "skill_version": self.skill_version,
            "success": self.success,
            "parameters": _json_safe(self.parameters),
            "scene_summary": _json_safe(self.scene_summary),
            "skill_metrics": _json_safe(self.skill_metrics),
            "verification": _json_safe(self.verification),
            "trace_id": self.trace_id,
            "trace_path": self.trace_path,
            "experiment_id": self.experiment_id,
            "study_id": self.study_id,
            "failure_stage": self.failure_stage,
            "failure_codes": list(self.failure_codes),
            "artifact_refs": list(self.artifact_refs),
            "eligible_as_training_data": self.eligible_as_training_data,
            "tags": list(self.tags),
        }


class SkillMemory:
    """Filesystem-backed append-only collection of skill experiences."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.records_dir = self.root / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def append(self, experience: SkillExperience) -> Path:
        path = self.records_dir / f"{experience.experience_id}.json"
        _atomic_create(path, experience.to_dict())
        return path

    def list_records(
        self,
        *,
        task_id: str | None = None,
        skill_name: str | None = None,
        success: bool | None = None,
        training_eligible_only: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        for path in sorted(self.records_dir.glob("skill_exp_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if task_id is not None and payload.get("task_id") != task_id:
                continue
            if skill_name is not None and payload.get("skill_name") != skill_name:
                continue
            if success is not None and bool(payload.get("success")) is not success:
                continue
            if training_eligible_only and not payload.get(
                "eligible_as_training_data", False
            ):
                continue
            payload["memory_record_path"] = str(path)
            records.append(payload)
        return tuple(records)

    def failure_summary(
        self,
        *,
        task_id: str | None = None,
        skill_name: str | None = None,
    ) -> dict[str, Any]:
        failures = self.list_records(
            task_id=task_id,
            skill_name=skill_name,
            success=False,
        )
        stages = Counter(record.get("failure_stage") for record in failures)
        codes = Counter(
            code
            for record in failures
            for code in record.get("failure_codes", [])
        )
        return {
            "failure_count": len(failures),
            "by_stage": dict(sorted(stages.items())),
            "by_code": dict(sorted(codes.items())),
            "experience_ids": [record["experience_id"] for record in failures],
        }


@dataclass(frozen=True)
class EvolutionProposal:
    """One single-variable policy branch that still requires evaluation."""

    proposal_id: str
    created_at_utc: str
    task_id: str
    skill_name: str
    parent_experiment_id: str
    hypothesis: str
    base_parameters: Mapping[str, Any]
    parameter_changes: Mapping[str, Any]
    supporting_experience_ids: tuple[str, ...] = ()
    status: str = "proposed"

    def __post_init__(self) -> None:
        if self.status != "proposed":
            raise ValueError("new evolution proposals must have status='proposed'")
        for value in (
            self.proposal_id,
            self.task_id,
            self.skill_name,
            self.parent_experiment_id,
            self.hypothesis,
        ):
            if not str(value).strip():
                raise ValueError("evolution proposal text fields must not be empty")
        if len(self.parameter_changes) != 1:
            raise ValueError("each proposal must change exactly one parameter")
        if not set(self.parameter_changes).issubset(self.base_parameters):
            raise ValueError("proposal may change only an existing base parameter")
        _canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "proposal_id": self.proposal_id,
            "created_at_utc": self.created_at_utc,
            "task_id": self.task_id,
            "skill_name": self.skill_name,
            "parent_experiment_id": self.parent_experiment_id,
            "hypothesis": self.hypothesis,
            "base_parameters": _json_safe(self.base_parameters),
            "parameter_changes": _json_safe(self.parameter_changes),
            "supporting_experience_ids": list(self.supporting_experience_ids),
            "status": self.status,
            "requires_independent_evaluation": True,
            "automatically_promoted": False,
        }


class BoundedEvolutionProposer:
    """Enumerate only explicitly allowed, one-variable-at-a-time branches."""

    def __init__(
        self,
        *,
        search_space: Mapping[str, Sequence[Any]],
        maximum_proposals: int = 8,
    ) -> None:
        if maximum_proposals <= 0:
            raise ValueError("maximum_proposals must be positive")
        if not search_space:
            raise ValueError("bounded search space must not be empty")
        self.search_space = {
            str(name): tuple(values) for name, values in search_space.items()
        }
        self.maximum_proposals = int(maximum_proposals)
        for name, values in self.search_space.items():
            if not name or not values:
                raise ValueError("every evolution parameter needs candidate values")
            for value in values:
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise ValueError("evolution values must be finite")
                _canonical_json(value)

    def propose(
        self,
        *,
        task_id: str,
        skill_name: str,
        parent_experiment_id: str,
        base_parameters: Mapping[str, Any],
        failure_summary: Mapping[str, Any],
    ) -> tuple[EvolutionProposal, ...]:
        unknown = sorted(set(self.search_space) - set(base_parameters))
        if unknown:
            raise ValueError(
                "search space contains parameters absent from the base: "
                + ", ".join(unknown)
            )
        failure_codes = Counter(failure_summary.get("by_code", {}))
        dominant_code = (
            failure_codes.most_common(1)[0][0]
            if failure_codes
            else "unclassified_failure"
        )
        support_ids = tuple(failure_summary.get("experience_ids", ()))[:20]
        proposals: list[EvolutionProposal] = []
        for name in sorted(self.search_space):
            base_value = base_parameters[name]
            for candidate in self.search_space[name]:
                if _canonical_json(candidate) == _canonical_json(base_value):
                    continue
                timestamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S_%fZ"
                )
                proposals.append(
                    EvolutionProposal(
                        proposal_id=(
                            f"evo_{timestamp}_{uuid.uuid4().hex[:10]}"
                        ),
                        created_at_utc=_utc_now(),
                        task_id=task_id,
                        skill_name=skill_name,
                        parent_experiment_id=parent_experiment_id,
                        hypothesis=(
                            f"Changing only {name} from {base_value!r} to "
                            f"{candidate!r} may reduce {dominant_code}."
                        ),
                        base_parameters=dict(base_parameters),
                        parameter_changes={name: candidate},
                        supporting_experience_ids=support_ids,
                    )
                )
                if len(proposals) >= self.maximum_proposals:
                    return tuple(proposals)
        return tuple(proposals)


class EvolutionQueue:
    """Append-only queue; proposals cannot silently replace a policy."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.proposals_dir = self.root / "proposals"
        self.proposals_dir.mkdir(parents=True, exist_ok=True)

    def append(self, proposal: EvolutionProposal) -> Path:
        path = self.proposals_dir / f"{proposal.proposal_id}.json"
        _atomic_create(path, proposal.to_dict())
        return path

    def list_proposals(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.proposals_dir.glob("evo_*.json"))
        )
