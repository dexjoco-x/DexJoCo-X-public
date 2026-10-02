"""Versioned skill manifests and invocation validation for AutoCollect."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .core import _json_safe


class ParameterKind(str, Enum):
    """JSON-compatible parameter types accepted by a skill manifest."""

    ARRAY = "array"
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class SkillParameterSpec:
    """Validation rule for one public skill parameter."""

    name: str
    kind: ParameterKind
    description: str
    required: bool = True
    shape: tuple[int, ...] | None = None
    finite: bool = True
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("skill parameter name and description must not be empty")
        if self.shape is not None:
            if self.kind is not ParameterKind.ARRAY:
                raise ValueError("only array parameters may declare a shape")
            if not self.shape or any(int(size) <= 0 for size in self.shape):
                raise ValueError("array parameter shape must contain positive sizes")
        if self.minimum is not None and not np.isfinite(self.minimum):
            raise ValueError("parameter minimum must be finite")
        if self.maximum is not None and not np.isfinite(self.maximum):
            raise ValueError("parameter maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("parameter minimum must not exceed maximum")
        if self.choices and self.kind is not ParameterKind.STRING:
            raise ValueError("only string parameters may declare choices")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("parameter choices must be unique")

    def validate(self, value: Any) -> Any:
        """Validate and normalize one parameter value."""

        if self.kind is ParameterKind.ARRAY:
            array = np.asarray(value, dtype=np.float64)
            if self.shape is not None and array.shape != self.shape:
                raise ValueError(
                    f"parameter {self.name!r} has shape {array.shape}; "
                    f"expected {self.shape}"
                )
            if self.finite and not np.all(np.isfinite(array)):
                raise ValueError(f"parameter {self.name!r} contains NaN or Inf")
            if self.minimum is not None and np.any(array < self.minimum):
                raise ValueError(
                    f"parameter {self.name!r} is below {self.minimum}"
                )
            if self.maximum is not None and np.any(array > self.maximum):
                raise ValueError(
                    f"parameter {self.name!r} exceeds {self.maximum}"
                )
            return array

        if self.kind is ParameterKind.STRING:
            if not isinstance(value, str):
                raise TypeError(f"parameter {self.name!r} must be a string")
            if self.choices and value not in self.choices:
                raise ValueError(
                    f"parameter {self.name!r} must be one of {self.choices}"
                )
            return value

        if self.kind is ParameterKind.BOOLEAN:
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"parameter {self.name!r} must be boolean")
            return bool(value)

        if self.kind is ParameterKind.INTEGER:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"parameter {self.name!r} must be an integer")
            normalized: int | float = int(value)
        elif self.kind is ParameterKind.NUMBER:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"parameter {self.name!r} must be numeric")
            normalized = float(value)
            if self.finite and not np.isfinite(normalized):
                raise ValueError(f"parameter {self.name!r} must be finite")
        else:
            raise RuntimeError(f"unsupported parameter kind: {self.kind}")

        if self.minimum is not None and normalized < self.minimum:
            raise ValueError(f"parameter {self.name!r} is below {self.minimum}")
        if self.maximum is not None and normalized > self.maximum:
            raise ValueError(f"parameter {self.name!r} exceeds {self.maximum}")
        return normalized

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "description": self.description,
            "required": self.required,
            "shape": list(self.shape) if self.shape is not None else None,
            "finite": self.finite,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "choices": list(self.choices),
        }


@dataclass(frozen=True)
class SkillManifest:
    """Machine-readable contract exposed to a future planner."""

    name: str
    version: str
    description: str
    supported_tasks: tuple[str, ...]
    parameters: tuple[SkillParameterSpec, ...]
    required_observations: tuple[str, ...]
    action_dimension: int
    required_perception: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    expected_effects: tuple[str, ...] = ()
    safety_constraints: tuple[str, ...] = ()
    implementation: str = ""
    controller_family: str = "deterministic"
    vla_used: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for label, value in (
            ("name", self.name),
            ("version", self.version),
            ("description", self.description),
            ("implementation", self.implementation),
            ("controller_family", self.controller_family),
        ):
            if not value.strip():
                raise ValueError(f"skill manifest {label} must not be empty")
        if not self.supported_tasks or any(
            not task.strip() for task in self.supported_tasks
        ):
            raise ValueError("skill manifest must support at least one named task")
        if len(set(self.supported_tasks)) != len(self.supported_tasks):
            raise ValueError("supported task names must be unique")
        parameter_names = [parameter.name for parameter in self.parameters]
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("skill parameter names must be unique")
        if not self.required_observations:
            raise ValueError("skill manifest must declare required observations")
        if int(self.action_dimension) <= 0:
            raise ValueError("skill manifest action dimension must be positive")

    def validate_invocation(
        self,
        *,
        task_id: str,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reject unsupported tasks, missing values, and undeclared values."""

        if task_id not in self.supported_tasks:
            raise ValueError(
                f"skill {self.name!r} does not support task {task_id!r}; "
                f"supported: {', '.join(self.supported_tasks)}"
            )
        supplied = set(parameters)
        declared = {parameter.name for parameter in self.parameters}
        unknown = sorted(supplied - declared)
        if unknown:
            raise ValueError(
                f"skill {self.name!r} received undeclared parameters: "
                f"{', '.join(unknown)}"
            )
        missing = sorted(
            parameter.name
            for parameter in self.parameters
            if parameter.required and parameter.name not in supplied
        )
        if missing:
            raise ValueError(
                f"skill {self.name!r} is missing required parameters: "
                f"{', '.join(missing)}"
            )
        return {
            parameter.name: parameter.validate(parameters[parameter.name])
            for parameter in self.parameters
            if parameter.name in supplied
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "supported_tasks": list(self.supported_tasks),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "required_observations": list(self.required_observations),
            "action_dimension": int(self.action_dimension),
            "required_perception": list(self.required_perception),
            "preconditions": list(self.preconditions),
            "expected_effects": list(self.expected_effects),
            "safety_constraints": list(self.safety_constraints),
            "implementation": self.implementation,
            "controller_family": self.controller_family,
            "vla_used": self.vla_used,
            "tags": list(self.tags),
        }

    def json_safe(self) -> dict[str, Any]:
        """Return an explicitly JSON-safe manifest representation."""

        return _json_safe(self.to_dict())
