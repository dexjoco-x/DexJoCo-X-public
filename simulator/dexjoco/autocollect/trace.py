"""Structured per-primitive traces for diagnosis and agent feedback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import uuid

from .core import _json_safe


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceStatus(str, Enum):
    """Terminal state of one trace event."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class TraceEvent:
    """One auditable environment, perception, skill, or storage operation."""

    sequence: int
    event_id: str
    trace_id: str
    attempt_id: str
    name: str
    category: str
    status: TraceStatus
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    start_step: int
    end_step: int
    experiment_id: str | None = None
    study_id: str | None = None
    parent_event_id: str | None = None
    inputs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    failure_code: str | None = None
    reason: str | None = None
    artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("trace sequence must be non-negative")
        if not self.event_id or not self.trace_id or not self.attempt_id:
            raise ValueError("trace, attempt, and event IDs must not be empty")
        if not self.name or not self.category:
            raise ValueError("trace event name and category must not be empty")
        if self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("trace event step interval is invalid")
        if self.duration_seconds < 0.0:
            raise ValueError("trace duration must be non-negative")
        if self.status is TraceStatus.FAILED:
            if not self.failure_code or not self.reason:
                raise ValueError("a failed trace event requires code and reason")
        elif self.failure_code is not None:
            raise ValueError("only a failed trace event may have a failure code")
        if self.status is TraceStatus.SUCCEEDED and self.reason is not None:
            raise ValueError("a successful trace event cannot have a reason")
        if self.status is TraceStatus.SKIPPED and not self.reason:
            raise ValueError("a skipped trace event requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "sequence": self.sequence,
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "attempt_id": self.attempt_id,
            "experiment_id": self.experiment_id,
            "study_id": self.study_id,
            "parent_event_id": self.parent_event_id,
            "name": self.name,
            "category": self.category,
            "status": self.status.value,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "duration_seconds": self.duration_seconds,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "inputs": _json_safe(self.inputs),
            "outputs": _json_safe(self.outputs),
            "metrics": _json_safe(self.metrics),
            "failure_code": self.failure_code,
            "reason": self.reason,
            "artifact_refs": list(self.artifact_refs),
        }


class TraceSpan:
    """Context manager that guarantees a terminal event on exceptions."""

    def __init__(
        self,
        recorder: "TraceRecorder",
        *,
        sequence: int,
        event_id: str,
        parent_event_id: str | None,
        name: str,
        category: str,
        start_step: int,
        inputs: Mapping[str, Any],
    ) -> None:
        self._recorder = recorder
        self.sequence = sequence
        self.event_id = event_id
        self.parent_event_id = parent_event_id
        self.name = name
        self.category = category
        self.start_step = int(start_step)
        self.inputs = dict(inputs)
        self._started_at_utc = _utc_now()
        self._started = time.perf_counter()
        self._finalized = False
        self._entered = False

    def __enter__(self) -> "TraceSpan":
        if self._entered:
            raise RuntimeError("a trace span cannot be entered twice")
        self._entered = True
        self._recorder._active_event_ids.append(self.event_id)
        return self

    def _finalize(
        self,
        *,
        status: TraceStatus,
        end_step: int,
        outputs: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        failure_code: str | None = None,
        reason: str | None = None,
        artifact_refs: tuple[str, ...] = (),
    ) -> None:
        if self._finalized:
            raise RuntimeError(f"trace event {self.name!r} is already terminal")
        event = TraceEvent(
            sequence=self.sequence,
            event_id=self.event_id,
            trace_id=self._recorder.trace_id,
            attempt_id=self._recorder.attempt_id,
            experiment_id=self._recorder.experiment_id,
            study_id=self._recorder.study_id,
            parent_event_id=self.parent_event_id,
            name=self.name,
            category=self.category,
            status=status,
            started_at_utc=self._started_at_utc,
            finished_at_utc=_utc_now(),
            duration_seconds=time.perf_counter() - self._started,
            start_step=self.start_step,
            end_step=int(end_step),
            inputs=self.inputs,
            outputs=dict(outputs or {}),
            metrics=dict(metrics or {}),
            failure_code=failure_code,
            reason=reason,
            artifact_refs=tuple(artifact_refs),
        )
        self._recorder._events.append(event)
        self._finalized = True

    def succeed(
        self,
        *,
        end_step: int,
        outputs: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifact_refs: tuple[str, ...] = (),
    ) -> None:
        self._finalize(
            status=TraceStatus.SUCCEEDED,
            end_step=end_step,
            outputs=outputs,
            metrics=metrics,
            artifact_refs=artifact_refs,
        )

    def fail(
        self,
        *,
        end_step: int,
        failure_code: str,
        reason: str,
        outputs: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifact_refs: tuple[str, ...] = (),
    ) -> None:
        self._finalize(
            status=TraceStatus.FAILED,
            end_step=end_step,
            outputs=outputs,
            metrics=metrics,
            failure_code=failure_code,
            reason=reason,
            artifact_refs=artifact_refs,
        )

    def skip(
        self,
        *,
        end_step: int,
        reason: str,
        outputs: Mapping[str, Any] | None = None,
    ) -> None:
        self._finalize(
            status=TraceStatus.SKIPPED,
            end_step=end_step,
            outputs=outputs,
            reason=reason,
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del traceback
        if not self._entered:
            raise RuntimeError("trace span exited without being entered")
        active = self._recorder._active_event_ids.pop()
        if active != self.event_id:
            raise RuntimeError("trace span nesting is corrupted")
        if exc is not None and not self._finalized:
            self.fail(
                end_step=self.start_step,
                failure_code="exception",
                reason=f"{exc_type.__name__}: {exc}",
                metrics={"exception_type": exc_type.__name__},
            )
        elif not self._finalized:
            self.succeed(end_step=self.start_step)
        return False


class TraceRecorder:
    """Collect ordered trace events for one candidate attempt."""

    def __init__(
        self,
        *,
        attempt_id: str,
        experiment_id: str | None = None,
        study_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        attempt_id = str(attempt_id).strip()
        if not attempt_id:
            raise ValueError("attempt_id must not be empty")
        self.attempt_id = attempt_id
        self.experiment_id = experiment_id
        self.study_id = study_id
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex}"
        self._events: list[TraceEvent] = []
        self._active_event_ids: list[str] = []
        self._next_sequence = 0

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(sorted(self._events, key=lambda event: event.sequence))

    def span(
        self,
        name: str,
        *,
        category: str,
        start_step: int,
        inputs: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        name = str(name).strip()
        category = str(category).strip()
        if not name or not category:
            raise ValueError("trace span name and category must not be empty")
        sequence = self._next_sequence
        self._next_sequence += 1
        parent = self._active_event_ids[-1] if self._active_event_ids else None
        return TraceSpan(
            self,
            sequence=sequence,
            event_id=f"{self.trace_id}_event_{sequence:04d}",
            parent_event_id=parent,
            name=name,
            category=category,
            start_step=start_step,
            inputs=dict(inputs or {}),
        )

    def summary(self) -> dict[str, Any]:
        events = self.events
        return {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "attempt_id": self.attempt_id,
            "experiment_id": self.experiment_id,
            "study_id": self.study_id,
            "event_count": len(events),
            "failed_event_count": sum(
                event.status is TraceStatus.FAILED for event in events
            ),
            "skipped_event_count": sum(
                event.status is TraceStatus.SKIPPED for event in events
            ),
            "event_names": [event.name for event in events],
        }

    def write_jsonl(self, path: str | Path) -> Path:
        """Atomically save all terminal events in sequence order."""

        if self._active_event_ids:
            raise RuntimeError("cannot write a trace with active spans")
        if not self._events:
            raise ValueError("cannot write an empty trace")
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}_",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for event in self.events:
                    json.dump(
                        event.to_dict(),
                        stream,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return path
