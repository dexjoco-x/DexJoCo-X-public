"""Explicit training schema: no simulator object truth in policy observations.

Rows are right then left. State is TCP xyz+quaternion(wxyz), then every native
hand joint (including passive/mimic joints). Action is world delta xyz+rotvec,
then native position-actuator targets. Padding is ALWAYS accompanied by masks.
"""
import numpy as np
from ..autocollect.retarget import _matrix_to_pose7

SCHEMA_VERSION="benchmark6_native_masked_v2"
MAX_HAND_JOINTS=24
MAX_HAND_ACTUATORS=22
SIDES=("right","left")


def public_observation(env):
    state=np.zeros((2,7+MAX_HAND_JOINTS),dtype=np.float32)
    mask=np.zeros(state.shape,dtype=bool)
    for row,side in enumerate(SIDES):
        if side not in env._full_hand_qpos:continue
        q=env._data.qpos[env._full_hand_qpos[side]]
        if len(q)>MAX_HAND_JOINTS:raise ValueError("Native state exceeds schema capacity")
        state[row,:7]=_matrix_to_pose7(env.tcp(side))
        state[row,7:7+len(q)]=q
        mask[row,:7+len(q)]=True
    return dict(state=state,state_mask=mask,arm_mask=mask[:,0].copy())


def pack_action(env,action):
    if not isinstance(action,dict):
        if len(env._native_ctrl_ids)!=1:raise ValueError("Dual action requires named arms")
        action={"right":action}
    if set(action)!=set(env._native_ctrl_ids):raise ValueError("Wrong active arm set")
    values=np.zeros((2,6+MAX_HAND_ACTUATORS),dtype=np.float64)
    mask=np.zeros(values.shape,dtype=bool)
    for row,side in enumerate(SIDES):
        if side not in action:continue
        value=np.asarray(action[side],dtype=np.float64)
        n=6+len(env._native_ctrl_ids[side])
        if n>values.shape[1]:raise ValueError("Native action exceeds schema capacity")
        if value.shape!=(n,) or not np.isfinite(value).all():raise ValueError("Invalid native action")
        values[row,:n]=value;mask[row,:n]=True
    return dict(action=values,action_mask=mask)


def unpack_action(env,values,mask):
    values=np.asarray(values);mask=np.asarray(mask,dtype=bool)
    if values.shape!=(2,6+MAX_HAND_ACTUATORS) or mask.shape!=values.shape:raise ValueError("Wrong padded action shape")
    if not np.isfinite(values).all():raise ValueError("Non-finite padded action")
    expected=pack_action(env,env.hold_action())["action_mask"]
    if not np.array_equal(mask,expected):raise ValueError("Action mask disagrees with native embodiment")
    if np.any(values[~mask]!=0):raise ValueError("Nonzero action padding")
    commands = {side:values[row,:6+len(env._native_ctrl_ids[side])].copy()
                for row,side in enumerate(SIDES) if side in env._native_ctrl_ids}
    return commands if len(commands) > 1 else commands["right"]


def schema_metadata(env):
    return dict(version=SCHEMA_VERSION,sides=list(SIDES),state_shape=[2,7+MAX_HAND_JOINTS],action_shape=[2,6+MAX_HAND_ACTUATORS],
                state="TCP xyz + quaternion wxyz + all native hand joint positions",
                action="world-frame delta xyz + rotation vector + native position-actuator targets",
                state_padding="zero, masked",action_padding="zero, masked",
                control_dt=env.contract["control_dt"],privileged_object_state_in_policy_inputs=False,
                joint_names={side:spec["state_joint_names"] for side,spec in env.hand_metadata["sides"].items()},
                actuator_names={side:spec["actuator_names"] for side,spec in env.hand_metadata["sides"].items()})
