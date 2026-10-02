"""Second task adapter for the same guarded chain and post-stage interfaces.

The source segment already contains demonstrated grasp/lift motion; it is
not an isolated approach-only or verified grasp skill.
"""
from dataclasses import replace
import hashlib
import json

import numpy as np

from .chain import BoundStageHandoff, ChainStage, SkillChainContract, SkillHandoffError
from .retarget import ObjectFrameRetargetSkill, make_object_frame_retarget_manifest


BUCKET_CHAIN = SkillChainContract("pick_bucket.phase_chain", (
    ChainStage("source_grasp_lift_motion", ("pick_bucket.source_grasp_lift_motion",)),
    ChainStage("grip_settle_motion", ("pick_bucket.grip_settle_motion",)),
    ChainStage("extra_lift_and_hold", ("pick_bucket.extra_lift_and_hold",)),
))


def make_bucket_phase_manifests(*, task_id, expected_action_dim):
    base = make_object_frame_retarget_manifest(task_id=task_id, expected_action_dim=expected_action_dim)
    return tuple(replace(base, name=stage.allowed_skills[0],
        description=f"Execute {stage.stage_id} from one bound bucket candidate; no physical grasp claim.",
        expected_effects=("declared motion segment completed, physical object effect unverified",),
        preconditions=base.preconditions + ("same candidate, locked reset perception and preceding motion receipt",),
        implementation="dexjoco.autocollect.bucket_chain.BucketPhaseSkill",
        tags=base.tags + ("guarded_phase_chain", "physical_effect_unverified"),
    ) for stage in BUCKET_CHAIN.stages)


class BucketPhaseSkill(ObjectFrameRetargetSkill):
    def __init__(self, candidate, *, stage_index, handoff):
        super().__init__(candidate)
        self.stage_index = stage_index
        self.handoff = handoff
        self.name = BUCKET_CHAIN.stages[stage_index].allowed_skills[0]

    def _enabled_phases(self):
        return ({"source"}, {"settle"}, {"lift", "hold"})[self.stage_index]

    @staticmethod
    def _checkpoint(context):
        state = np.asarray(context.observation["state"], dtype=np.float64).reshape(-1)[:13]
        if state.shape != (13,) or not np.all(np.isfinite(state)):
            raise SkillHandoffError("bucket handoff requires TCP and hand proprioception")
        return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()

    def execute(self, context, parameters):
        try:
            signature = json.dumps({"candidate_binding": id(self.candidate),
                "bucket": np.asarray(parameters["target_bucket_pose7"], dtype=float).tolist(),
                "source": parameters["target_pose_source"],
                "controller": {key: value for key, value in vars(self).items()
                               if key not in {"candidate", "stage_index", "handoff", "name"}},
            }, sort_keys=True, allow_nan=False)
            self.handoff.enter(self.stage_index, context, signature=signature, checkpoint=self._checkpoint(context))
            result = super().execute(context, parameters)
            checkpoint = self._checkpoint(context)
            self.handoff.finish(context, success=result.success, checkpoint=checkpoint)
            return replace(result, metrics={**result.metrics,
                "chain_id": BUCKET_CHAIN.chain_id, "stage_id": BUCKET_CHAIN.stages[self.stage_index].stage_id,
                "stage_index": self.stage_index, "physical_effect_verified": False,
                "handoff_checkpoint_sha256": checkpoint})
        except Exception:
            self.handoff.invalidate()
            raise


def bind_bucket_phase_chain(candidate):
    handoff = BoundStageHandoff(3)
    return tuple(BucketPhaseSkill(candidate, stage_index=index, handoff=handoff) for index in range(3))
