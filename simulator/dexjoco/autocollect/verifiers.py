"""Task verifiers kept outside the policy or atomic-skill implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import _json_safe


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "metrics": _json_safe(self.metrics),
        }


class PickBucketLiftVerifier:
    """Verify that the payload stays inside a bucket lifted from reset height."""

    def __init__(self, minimum_lift_m: float = 0.15) -> None:
        minimum_lift_m = float(minimum_lift_m)
        if not np.isfinite(minimum_lift_m) or minimum_lift_m <= 0.0:
            raise ValueError("minimum_lift_m must be finite and positive")
        self.minimum_lift_m = minimum_lift_m

    def verify(self, env: Any) -> VerificationResult:
        raw_env = env.unwrapped
        box_position = np.asarray(
            raw_env._data.sensor("boxed_food_0_pos").data,
            dtype=np.float64,
        )
        site_ids = [
            int(raw_env.model.site(f"bucket_ref_{index}").id)
            for index in range(8)
        ]
        corners = np.asarray(raw_env._data.site_xpos[site_ids], dtype=np.float64)
        inside = bool(
            np.all(box_position >= corners.min(axis=0))
            and np.all(box_position <= corners.max(axis=0))
        )
        bottom_z = np.asarray(
            raw_env._data.site_xpos[raw_env._bucket_bottom_site_ids, 2],
            dtype=np.float64,
        )
        baseline_z = np.asarray(raw_env._bucket_bottom_z0, dtype=np.float64)
        lift_per_reference = bottom_z - baseline_z
        minimum_lift = float(np.min(lift_per_reference))
        lifted = bool(np.all(lift_per_reference >= self.minimum_lift_m))
        success = inside and lifted
        if not inside:
            reason = "payload is outside the bucket"
        elif not lifted:
            reason = (
                f"minimum bucket lift {minimum_lift:.4f} m is below "
                f"{self.minimum_lift_m:.4f} m"
            )
        else:
            reason = None
        return VerificationResult(
            success=success,
            reason=reason,
            metrics={
                "boxed_food_inside": inside,
                "minimum_bucket_lift_m": minimum_lift,
                "required_bucket_lift_m": self.minimum_lift_m,
                "all_bucket_references_lifted": lifted,
            },
        )


class HammerNailDepthVerifier:
    """Verify nail insertion depth without using the result to choose actions."""

    def __init__(self, minimum_depth_m: float = 0.04) -> None:
        minimum_depth_m = float(minimum_depth_m)
        if not np.isfinite(minimum_depth_m) or minimum_depth_m <= 0.0:
            raise ValueError("minimum_depth_m must be finite and positive")
        self.minimum_depth_m = minimum_depth_m

    def verify(self, env: Any) -> VerificationResult:
        raw_env = env.unwrapped
        depth = float(raw_env.get_nail_depth())
        if not np.isfinite(depth):
            return VerificationResult(
                success=False,
                reason="nail insertion depth is not finite",
                metrics={
                    "nail_insertion_depth_m": depth,
                    "required_nail_insertion_depth_m": self.minimum_depth_m,
                },
            )
        success = depth >= self.minimum_depth_m
        reason = None
        if not success:
            reason = (
                f"nail insertion depth {depth:.4f} m is below "
                f"{self.minimum_depth_m:.4f} m"
            )
        return VerificationResult(
            success=success,
            reason=reason,
            metrics={
                "nail_insertion_depth_m": depth,
                "required_nail_insertion_depth_m": self.minimum_depth_m,
                "depth_threshold_reached": success,
            },
        )
