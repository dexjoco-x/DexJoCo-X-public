"""Crash-resistant experiment ledger for AutoCollect autoresearch loops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

from .core import _json_safe


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def configuration_sha256(configuration: Mapping[str, Any]) -> str:
    """Return a stable hash for policy or experiment configuration."""

    return hashlib.sha256(_canonical_json(configuration)).hexdigest()


def source_sha256(objects_or_paths: Sequence[Any]) -> tuple[str, tuple[str, ...]]:
    """Hash the source files defining a policy implementation."""

    paths: set[Path] = set()
    for value in objects_or_paths:
        if isinstance(value, (str, Path)):
            path = Path(value)
        else:
            source = inspect.getsourcefile(value)
            if source is None:
                raise ValueError(f"cannot locate source for {value!r}")
            path = Path(source)
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"policy source file not found: {path}")
        paths.add(path)
    if not paths:
        raise ValueError("at least one policy source is required")

    digest = hashlib.sha256()
    ordered = tuple(sorted(str(path) for path in paths))
    for path_text in ordered:
        path = Path(path_text)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), ordered


@dataclass(frozen=True)
class PolicyIdentity:
    """Auditable identity of the policy tested by one experiment."""

    name: str
    version: str
    code_sha256: str
    config_sha256: str
    source_files: tuple[str, ...] = ()

    @classmethod
    def from_sources(
        cls,
        *,
        name: str,
        version: str,
        sources: Sequence[Any],
        configuration: Mapping[str, Any],
    ) -> "PolicyIdentity":
        code_hash, source_files = source_sha256(sources)
        return cls(
            name=name,
            version=version,
            code_sha256=code_hash,
            config_sha256=configuration_sha256(configuration),
            source_files=source_files,
        )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("policy name must not be empty")
        if not self.version.strip():
            raise ValueError("policy version must not be empty")
        for name, value in (
            ("code_sha256", self.code_sha256),
            ("config_sha256", self.config_sha256),
        ):
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "source_files": list(self.source_files),
        }


@dataclass(frozen=True)
class EnvironmentIdentity(PolicyIdentity):
    """Versioned identity for reset, safety, and verification code."""


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable declaration made before an experiment starts."""

    task_id: str
    seed: int
    policy: PolicyIdentity
    hypothesis: str
    environment: EnvironmentIdentity | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    parent_experiment_id: str | None = None
    study_id: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.hypothesis.strip():
            raise ValueError("hypothesis must not be empty")
        if (
            self.parent_experiment_id is not None
            and not self.parent_experiment_id.strip()
        ):
            raise ValueError("parent_experiment_id must be None or non-empty")
        if self.study_id is not None and not self.study_id.strip():
            raise ValueError("study_id must be None or non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": int(self.seed),
            "policy": self.policy.to_dict(),
            "environment": (
                self.environment.to_dict() if self.environment is not None else None
            ),
            "hypothesis": self.hypothesis,
            "parameters": _json_safe(self.parameters),
            "parent_experiment_id": self.parent_experiment_id,
            "study_id": self.study_id,
            "tags": list(self.tags),
        }


class ExperimentStatus(str, Enum):
    RUNNING = "running"
    PASSED = "pass"
    FAILED = "fail"
    ERROR = "error"


@dataclass(frozen=True)
class ExperimentHandle:
    experiment_id: str
    record_path: Path


class ExperimentLedger:
    """Store one atomically updated JSON record per experiment.

    Separate files avoid mixing results from concurrent workers.  Records may
    transition exactly once from ``running`` to ``pass``, ``fail``, or
    ``error``; declared parameters and policy fingerprints never change.
    """

    schema_version = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.records_dir = self.root / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def _record_path(self, experiment_id: str) -> Path:
        if not experiment_id or any(char in experiment_id for char in "/\\"):
            raise ValueError("invalid experiment_id")
        return self.records_dir / f"{experiment_id}.json"

    @staticmethod
    def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}_",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(_json_safe(payload), stream, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def start(
        self,
        spec: ExperimentSpec,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentHandle:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        experiment_id = f"exp_{timestamp}_{uuid.uuid4().hex[:10]}"
        path = self._record_path(experiment_id)
        record = {
            "schema_version": self.schema_version,
            "experiment_id": experiment_id,
            "status": ExperimentStatus.RUNNING.value,
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "spec": spec.to_dict(),
            "metadata": _json_safe(metadata or {}),
            "result": None,
            "error": None,
        }
        self._atomic_write(path, record)
        return ExperimentHandle(experiment_id=experiment_id, record_path=path)

    def load(self, experiment_id: str) -> dict[str, Any]:
        path = self._record_path(experiment_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def finish(
        self,
        handle: ExperimentHandle,
        *,
        status: ExperimentStatus,
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> Path:
        if status is ExperimentStatus.RUNNING:
            raise ValueError("finish requires a terminal experiment status")
        expected_path = self._record_path(handle.experiment_id)
        if handle.record_path.resolve() != expected_path:
            raise ValueError("experiment handle does not belong to this ledger")
        record = self.load(handle.experiment_id)
        if record["status"] != ExperimentStatus.RUNNING.value:
            raise RuntimeError(
                f"experiment {handle.experiment_id} is already terminal: "
                f"{record['status']}"
            )
        if status is ExperimentStatus.ERROR and not error:
            raise ValueError("an error experiment requires structured error details")
        if status is not ExperimentStatus.ERROR and error is not None:
            raise ValueError("only error experiments may contain error details")
        record["status"] = status.value
        record["finished_at_utc"] = _utc_now()
        record["result"] = _json_safe(result) if result is not None else None
        record["error"] = _json_safe(error) if error is not None else None
        self._atomic_write(expected_path, record)
        return expected_path

    def list_records(self) -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.records_dir.glob("exp_*.json"))
        ]
