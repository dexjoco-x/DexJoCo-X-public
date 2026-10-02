"""A guarded three-motion hammer chain; not an independent grasp detector.

All stages use the SAME retrieved demonstration and reset perception. These
are composable harness calls, not independently learned/reusable manipulation
policies. A fresh chain binding is required after any reset or failed stage.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np

from .chain import BoundStageHandoff, ChainStage, SkillChainContract, SkillHandoffError
from .hammer_retarget import (
    TwoFrameHammerRetargetSkill, infer_hammer_phase_plan,
    make_hammer_frame_retarget_manifest,
)


HAMMER_CHAIN = SkillChainContract(
    chain_id="hammer_nail.phase_chain",
    stages=(
        ChainStage("acquire_motion", ("hammer_nail.acquire_motion",)),
        ChainStage("frame_transition", ("hammer_nail.frame_transition",)),
        ChainStage("transport_and_strike", ("hammer_nail.transport_and_strike",)),
    ),
)


def make_hammer_phase_manifests(*, task_id, expected_action_dim):
    base = make_hammer_frame_retarget_manifest(
        task_id=task_id, expected_action_dim=expected_action_dim,
    )
    descriptions = (
        "Execute the source approach/hand-closing motion in the perceived hammer frame.",
        "Execute the source motion while blending from the hammer to the nail frame.",
        "Execute the source transport and strike motion in the perceived nail frame.",
    )
    return tuple(replace(
        base, name=stage.allowed_skills[0],
        description=description + " Motion completion does not certify physical grasp or task success.",
        preconditions=base.preconditions + (
            "all stages are bound to the same candidate and locked reset perception",
            "the immediately preceding motion receipt is required except for stage zero",
            "proprioceptive checkpoint, context identity and step counter must be unchanged at handoff",
        ),
        expected_effects=("the declared source motion segment is fully executed",),
        implementation="dexjoco.autocollect.hammer_chain.HammerPhaseSkill",
        tags=base.tags + ("guarded_phase_chain", "physical_effect_unverified"),
    ) for stage, description in zip(HAMMER_CHAIN.stages, descriptions))


class HammerPhaseSkill(TwoFrameHammerRetargetSkill):
    def __init__(self, candidate, *, stage_index, handoff, **controller_options):
        super().__init__(candidate, **controller_options)
        if type(stage_index) is not int or not 0 <= stage_index < len(HAMMER_CHAIN.stages):
            raise ValueError("invalid hammer chain stage index")
        self.stage_index = stage_index
        self.handoff = handoff
        self.name = HAMMER_CHAIN.stages[stage_index].allowed_skills[0]
        plan = infer_hammer_phase_plan(
            candidate.episode, grasp_settle_steps=self.grasp_settle_steps,
            frame_blend_steps=self.frame_blend_steps,
        )
        if any(end <= begin for _, begin, end, _ in super()._phase_ranges(plan)):
            raise ValueError("phase-chain candidates must contain three nonempty motion ranges")

    def _phase_ranges(self, plan):
        return (super()._phase_ranges(plan)[self.stage_index],)

    @staticmethod
    def _checkpoint(context):
        # Never use the state vector's reset object-pose fields (indices 13+).
        state = np.asarray(context.observation["state"], dtype=np.float64).reshape(-1)[:13]
        if state.shape != (13,) or not np.all(np.isfinite(state)):
            raise SkillHandoffError("handoff requires finite TCP and hand proprioception")
        return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()

    def execute(self, context, parameters):
        try:
            signature = json.dumps({
                "candidate_binding": id(self.candidate),
                "hammer": np.asarray(parameters["target_hammer_pose7"], dtype=float).tolist(),
                "nail": np.asarray(parameters["target_nail_pose7"], dtype=float).tolist(),
                "source": parameters["target_pose_source"],
                "controller": [self.translation_feedback_gain, self.rotation_feedback_gain,
                               self.grasp_settle_steps, self.frame_blend_steps],
            }, sort_keys=True, allow_nan=False)
            self.handoff.enter(self.stage_index, context, signature=signature,
                               checkpoint=self._checkpoint(context))
            result = super().execute(context, parameters)
            checkpoint = self._checkpoint(context)
            self.handoff.finish(context, success=result.success, checkpoint=checkpoint)
            return replace(result, metrics={
                **result.metrics,
                "chain_id": HAMMER_CHAIN.chain_id,
                "stage_id": HAMMER_CHAIN.stages[self.stage_index].stage_id,
                "stage_index": self.stage_index,
                "physical_effect_verified": False,
                "handoff_checkpoint_sha256": checkpoint,
                "handoff_next_stage_index": self.handoff.next_index if result.success else None,
            })
        except Exception:
            self.handoff.invalidate()
            raise


def bind_hammer_phase_chain(candidate, **controller_options):
    """Create fresh, mutually bound skills for exactly one candidate attempt."""
    handoff = BoundStageHandoff(len(HAMMER_CHAIN.stages))
    return tuple(HammerPhaseSkill(candidate, stage_index=index, handoff=handoff,
                                  **controller_options)
                 for index in range(len(HAMMER_CHAIN.stages)))
