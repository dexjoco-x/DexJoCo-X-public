"""Per-task helpers that patch a freshly reset env to match a recorded state[0].

`restore_initial_state` is the public entry point used by the replay script: it
splits the recorded flat proprio vector by `config.proprio_keys`, dispatches to
the task-specific restorer, runs an `mj_forward`, and returns the refreshed
wrapped observation.

Each restorer only writes the fields recorded as proprio (object poses + table
height). Everything else (lighting, table texture, third-person camera) is left
to whatever `env.reset()` produced, so visual randomization stays effective.
"""

'''
在 replay / 回放 demonstration 时，把刚 reset 出来的仿真环境，恢复到录制 demo 的初始状态 state[0]
所以这段代码会根据 demo 里保存的第一帧 proprio state，把桌子高度、物体位置、物体姿态等写回 MuJoCo 环境
'''

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

'''
它提供了一组 按任务分类的恢复函数。
water_plant               恢复喷壶、植物、桌子高度
click_mouse               恢复鼠标、鼠标垫、显示器高度
bimanual_assembly         恢复 socket、peg、桌子高度
bimanual_microwave_cook   恢复热狗、微波炉、桌子高度
'''

def _split_state_by_proprio(state_vec, proprio_keys, ref_state):
    """Split a flat proprio vector back into a {proprio_key: array} dict.

    Sizes come from a fresh `_compute_observation()` sample because some envs
    declare e.g. `gripper_pose` as `(1,)` in `observation_space` while actually
    returning a 16-d sensor vector; `DexjocoObsAdapter` flattens the real
    array, not the spec.
    """
    parts = {}
    offset = 0
    for key in proprio_keys:
        size = max(1, np.asarray(ref_state[key]).ravel().size)
        parts[key] = np.asarray(state_vec[offset : offset + size], dtype=np.float64)
        offset += size
    return parts


# 这个函数恢复 桌子高度随机化
def _apply_delta_h(raw_env, delta_h):
    """Mimic each env's table-height adjustment based on what attributes it caches."""
    raw_env.delta_h = np.float64(delta_h)
    z0 = getattr(raw_env, "_table_body_z0", None)
    if z0 is None:
        z0 = raw_env._table_z
    table_body_id = raw_env._model.body("table").id
    raw_env._model.body_pos[table_body_id, 2] = z0 + delta_h

    # Resolve which leg geoms to adjust:
    #   - most envs cache `_table_leg_geom_ids` (either hardcoded or discovered)
    #   - water_plant caches `_table_leg_half_len0` keyed by leg name instead
    #   - bimanual_microwave_cook caches neither and iterates 4 hardcoded names
    if hasattr(raw_env, "_table_leg_geom_ids"):
        leg_gids = list(raw_env._table_leg_geom_ids)
    else:
        h0_map = getattr(raw_env, "_table_leg_half_len0", None)
        if isinstance(h0_map, dict) and h0_map and isinstance(next(iter(h0_map)), str):
            leg_gids = [raw_env._model.geom(n).id for n in h0_map]
        else:
            leg_gids = [
                raw_env._model.geom(n).id
                for n in ("table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4")
            ]

    if hasattr(raw_env, "_model_geom_pos0"):
        # bimanual_microwave_cook / bimanual_unlock_ipad: shift center down and extend size by half.
        for gid in leg_gids:
            raw_env._model.geom_pos[gid, 2] = raw_env._model_geom_pos0[gid, 2] - 0.5 * delta_h
            raw_env._model.geom_size[gid, 1] = raw_env._model_geom_size0[gid, 1] + 0.5 * delta_h
        return

    # Standard style: extend each leg by the full delta.
    h0_map = raw_env._table_leg_half_len0
    for gid in leg_gids:
        if gid in h0_map:
            h0 = h0_map[gid]
        else:
            name = mujoco.mj_id2name(raw_env._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            h0 = h0_map[name]
        raw_env._model.geom_size[gid, 1] = h0 + delta_h

# 每个_restore_xxx 是一个任务专用恢复函数


def _restore_single_arm_recorded_controls(raw_env, parts):
    tcp = parts.get("tcp_pose")
    if tcp is not None and np.asarray(tcp).size >= 7:
        tcp = np.asarray(tcp, dtype=np.float64).reshape(-1)
        if getattr(raw_env._data, "mocap_pos", np.asarray([])).shape[0] > 0:
            raw_env._data.mocap_pos[0] = tcp[:3]
            quat = tcp[3:7].copy()
            norm = np.linalg.norm(quat)
            if norm > 1e-8:
                raw_env._data.mocap_quat[0] = quat / norm

    gripper = parts.get("gripper_pose")
    qpos_ids = getattr(raw_env, "_allegro_dof_ids", None)
    if gripper is None or qpos_ids is None:
        return

    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    qpos_ids = np.asarray(qpos_ids, dtype=int).reshape(-1)
    n = min(gripper.size, qpos_ids.size)
    if n <= 0:
        return

    raw_env._data.qpos[qpos_ids[:n]] = gripper[:n]
    vel_ids = getattr(raw_env, "_allegro_vel_ids", None)
    if vel_ids is not None:
        vel_ids = np.asarray(vel_ids, dtype=int).reshape(-1)
        raw_env._data.qvel[vel_ids[: min(n, vel_ids.size)]] = 0.0

    if hasattr(raw_env, "_hand_qpos_target"):
        raw_env._hand_qpos_target = gripper[:n].copy()

    ctrl_ids = getattr(raw_env, "_allegro_ctrl_ids", None)
    if ctrl_ids is not None and getattr(raw_env, "_hand_ctrl_matches_qpos", False):
        ctrl_ids = np.asarray(ctrl_ids, dtype=int).reshape(-1)
        m = min(n, ctrl_ids.size)
        valid = ctrl_ids[:m] >= 0
        if np.any(valid):
            ctrl_values = gripper[:m][valid]
            ranges = raw_env._model.actuator_ctrlrange[ctrl_ids[:m][valid]]
            ctrl_values = np.clip(ctrl_values, ranges[:, 0], ranges[:, 1])
            raw_env._data.ctrl[ctrl_ids[:m][valid]] = ctrl_values


def _refresh_recorded_table_height_caches(raw_env):
    mujoco.mj_forward(raw_env._model, raw_env._data)

    table_site_id = getattr(raw_env, "_table_site_id", None)
    if table_site_id is not None:
        try:
            raw_env._table_z = float(raw_env._data.site_xpos[int(table_site_id)][2])
        except Exception:
            pass

    if hasattr(raw_env, "_lift_height") and hasattr(raw_env, "_table_z"):
        raw_env._lift_z = float(raw_env._table_z) + float(raw_env._lift_height)
        update_lift_plane = getattr(raw_env, "_update_lift_plane_pose", None)
        if callable(update_lift_plane):
            update_lift_plane()


def _restore_water_plant(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    spray = parts["spray_ori_pose"]
    raw_env._data.jnt("spray_root").qpos[:3] = spray[:3]
    raw_env._data.jnt("spray_root").qpos[3:7] = spray[3:7]
    raw_env._spray_ori_pose = spray.copy()
    plant = parts["plant_ori_pose"]
    raw_env._model.body("plant").pos = plant[:3]
    raw_env._plant_ori_pose = plant.copy()


def _restore_click_mouse(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    mouse = parts["mouse_ori_pose"]
    raw_env._data.jnt("mouse_root").qpos = mouse
    raw_env.mouse_ori_pose = mouse.copy()
    # env.reset() ties the mousepad's xy to the freshly-sampled mouse_xy;
    # after restoring the mouse to the recorded pose, re-attach the
    # mousepad so the rigid-body contact graph matches.
    from ..sim.envs.panda_click_mouse_env import _MOUSEPAD_OFFSET
    mujoco.mj_forward(raw_env._model, raw_env._data)
    table_z = float(raw_env._data.site_xpos[raw_env._table_site_id][2])
    raw_env._model.body_pos[raw_env._mousepad_body_id][:3] = (
        np.array([mouse[0], mouse[1], table_z]) + _MOUSEPAD_OFFSET
    )
    # The monitor (display_root) has a free joint with `z = z0 + delta_h`;
    # reset() sampled this z with the original delta_h, but `_apply_delta_h`
    # above just rewrote `raw_env.delta_h` to the recorded value, leaving
    # the monitor floating or buried by `(delta_h_original - delta_h_recorded)`.
    # Its xy stays as whatever reset() sampled (not in proprio).
    raw_env._data.jnt("display_root").qpos[2] = (
        raw_env._display_root_z0 + raw_env.delta_h
    )


def _restore_pinch_tongs(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    tongs = parts["tongs_ori_pose"]
    raw_env._data.jnt("tongs_root").qpos = tongs
    raw_env.tongs_ori_pose = tongs.copy()


def _restore_pinch_tongs_oymotion(raw_env, parts):
    _restore_pinch_tongs(raw_env, parts)
    _refresh_recorded_table_height_caches(raw_env)
    _restore_single_arm_recorded_controls(raw_env, parts)
    keep_above_table = getattr(
        raw_env,
        "_keep_oymotion_wuji_tongs_above_table",
        None,
    )
    if callable(keep_above_table):
        keep_above_table()


def _restore_pinch_tongs_sharpa_wave(raw_env, parts):
    _restore_pinch_tongs(raw_env, parts)
    _refresh_recorded_table_height_caches(raw_env)
    _restore_single_arm_recorded_controls(raw_env, parts)
    keep_above_table = getattr(raw_env, "_keep_tongs_above_table", None)
    if callable(keep_above_table):
        keep_above_table()


def _restore_pinch_tongs_wuji(raw_env, parts):
    _restore_pinch_tongs(raw_env, parts)
    _refresh_recorded_table_height_caches(raw_env)
    _restore_single_arm_recorded_controls(raw_env, parts)
    keep_above_table = getattr(raw_env, "_keep_wuji_tongs_above_table", None)
    if callable(keep_above_table):
        keep_above_table()


def _restore_fold_glasses(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    glass = parts["glass_ori_pose"]
    raw_env._data.jnt("glass_root").qpos = glass
    raw_env.glass_ori_pose = glass.copy()
    box = parts["box_ori_pose"]
    box_body_id = raw_env._model.body("open_box").id
    raw_env._model.body_pos[box_body_id] = box[:3]
    raw_env._model.body_quat[box_body_id] = box[3:7]
    raw_env.open_box_ori_pose = box.copy()
    stop = parts.get("glass_stop_pose")
    if stop is not None and np.asarray(stop).size >= 7:
        try:
            stop_body_id = raw_env._model.body("glass_front_stop").id
        except Exception:
            stop_body_id = None
        if stop_body_id is not None:
            raw_env._model.body_pos[stop_body_id] = stop[:3]
            raw_env._model.body_quat[stop_body_id] = stop[3:7]
            raw_env.glass_stop_pose = stop[:7].copy()


def _restore_hammer_nail(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    raw_env._model.body_pos[raw_env._model.body("wood").id, 2] = (
        raw_env._wood_body_z0 + raw_env.delta_h
    )
    hammer = parts["hammer_ori_pose"]
    raw_env._data.jnt("hammer_joint").qpos = hammer
    raw_env.hammer_ori_pose = hammer.copy()
    nail = parts["nail_ori_pose"]
    nail_pos = nail[:3].copy()
    nail_pos[2] = raw_env._nail_body_z0 + raw_env.delta_h
    raw_env._nail_init_pos[:] = nail_pos
    raw_env._data.mocap_pos[raw_env._nail_mocap_id] = nail_pos
    raw_env._data.mocap_quat[raw_env._nail_mocap_id] = nail[3:7]
    raw_env.nail_ori_pose = np.concatenate([nail_pos, nail[3:7]])


def _restore_pick_bucket(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    bucket = parts["bucket_ori_pose"]
    raw_env._data.jnt("bucket_root").qpos = bucket
    raw_env.bucket_ori_pose = bucket.copy()
    raw_env._bucket_z = float(bucket[2])
    boxed_food = parts["boxed_food_ori_pose"]
    raw_env._data.jnt("boxed_food_0_freejoint").qpos = boxed_food
    raw_env.box_food_ori_pose = boxed_food.copy()
    # Re-baseline `_bucket_bottom_z0`: env.reset() captures it at the
    # randomly-sampled bucket pose, but success uses (bottom_z - z0 >= 0.15)
    # so the baseline must reflect the restored bucket pose.
    mujoco.mj_forward(raw_env._model, raw_env._data)
    raw_env._bucket_bottom_z0 = raw_env._data.site_xpos[
        raw_env._bucket_bottom_site_ids, 2
    ].copy()

# 他恢复了三样东西
# 1. 桌子高度 table_delta_height
# 2. socket 的初始位姿 socket_ori_pose
# 3. peg 的初始位姿 peg_ori_pose

def _restore_bimanual_assembly(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    socket = parts["socket_ori_pose"]
    raw_env._set_free_joint_pose(
        raw_env._socket_qpos_adr, raw_env._socket_qvel_adr, socket[:3], socket[3:7]
    )
    raw_env._socket_ori_pose = socket.copy()
    peg = parts["peg_ori_pose"]
    raw_env._set_free_joint_pose(
        raw_env._peg_qpos_adr, raw_env._peg_qvel_adr, peg[:3], peg[3:7]
    )
    raw_env._peg_ori_pose = peg.copy()


def _restore_bimanual_hanoi(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    base = np.asarray(parts["hanoi_base_ori_pos"], dtype=np.float64)
    raw_env._model.body_pos[raw_env._base_body_id] = base
    raw_env.base_ori_pos = base.copy()
    # Re-stack disks at the recorded base position; the tower preset matches
    # what reset() would pick, but we need to recompute it relative to the
    # restored base position (otherwise disks float above the prior location).
    base_delta_xy = base[:2] - raw_env._base_init_pos[:2]
    _, tower_state = raw_env._sample_reset_tower_state()
    raw_env._apply_reset_tower_state(tower_state, base_delta_xy)


def _restore_bimanual_microwave_cook(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    hot_dog = parts.get("hot_dog_ori_pose")
    if hot_dog is not None:
        raw_env._data.jnt("hot_dog_free").qpos = hot_dog
        raw_env._hot_dog_ori_pose = hot_dog.copy()
    microwave = parts.get("microwave_ori_pose")
    if microwave is not None:
        raw_env._model.body("microwave_object").pos = microwave[:3]
        raw_env._model.body("microwave_object").quat = microwave[3:7]
        raw_env._microwave_ori_pose = microwave.copy()


def _recover_leap_microwave_hot_dog_pose(raw_env, table_delta_height):
    """Recover legacy LEAP hot-dog reset state omitted from recorded proprio."""
    target_delta = float(table_delta_height)
    rng = np.random.RandomState(0)

    # LEAP microwave collection uses the environment's default seed=0. Each
    # reset draws table height, hot-dog yaw, then hot-dog x/y in this order.
    for _reset_index in range(1, 10001):
        sampled_delta = float(rng.uniform(0.0, 0.05))
        yaw_deg = float(rng.uniform(-20.0, 20.0))
        hot_dog_xy = np.asarray(
            rng.uniform([-0.35, -0.3], [-0.25, -0.4]),
            dtype=np.float64,
        )
        if not np.isclose(sampled_delta, target_delta, rtol=0.0, atol=1e-14):
            continue

        base_wxyz = np.asarray(
            raw_env._model.body("hot_dog").quat,
            dtype=np.float64,
        )
        base_xyzw = np.asarray(
            [base_wxyz[1], base_wxyz[2], base_wxyz[3], base_wxyz[0]],
            dtype=np.float64,
        )
        rotated = R.from_euler("z", yaw_deg, degrees=True) * R.from_quat(base_xyzw)
        quat_xyzw = rotated.as_quat()
        return np.asarray(
            [
                hot_dog_xy[0],
                hot_dog_xy[1],
                raw_env._hot_dog_body_z0 + target_delta,
                quat_xyzw[3],
                quat_xyzw[0],
                quat_xyzw[1],
                quat_xyzw[2],
            ],
            dtype=np.float64,
        )

    return None


def _restore_bimanual_microwave_cook_leap(raw_env, parts):
    """Restore LEAP microwave demos whose proprio omitted the hot-dog pose."""
    _restore_bimanual_microwave_cook(raw_env, parts)

    hot_dog = parts.get("hot_dog_ori_pose")
    if hot_dog is not None and np.asarray(hot_dog).size >= 7:
        return

    table_delta = np.asarray(parts.get("table_delta_height", ())).reshape(-1)
    if table_delta.size != 1:
        return

    recovered = _recover_leap_microwave_hot_dog_pose(raw_env, table_delta[0])
    if recovered is None:
        return

    raw_env._data.jnt("hot_dog_free").qpos = recovered
    raw_env._hot_dog_ori_pose = recovered.copy()


def _restore_bimanual_photograph(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    # The target_region geom is reset to (logo_pos + constant offset); preserve
    # that constant offset when we move the logo to the recorded pose.
    target_offset = (
        raw_env._model.geom_pos[raw_env._target_region_geom_id]
        - raw_env._model.geom_pos[raw_env._logo_geom_id]
    ).copy()
    logo = parts["logo_ori_pose"]
    raw_env._model.geom_pos[raw_env._logo_geom_id] = logo[:3]
    raw_env._model.geom_quat[raw_env._logo_geom_id] = logo[3:7]
    raw_env._model.geom_pos[raw_env._target_region_geom_id] = logo[:3] + target_offset
    raw_env.logo_ori_pose = logo.copy()
    camera = parts["camera_ori_pose"]
    raw_env._data.jnt("camera_root").qpos = camera
    raw_env.camera_ori_pose = camera.copy()


def _restore_bimanual_unlock_ipad(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    stand = parts["stand_ori_pose"]
    stand_body_id = raw_env._model.body("ipad_stand").id
    raw_env._model.body_pos[stand_body_id] = stand[:3]
    raw_env._stand_ori_pose = stand.copy()
    ipad = parts["ipad_ori_pose"]
    raw_env._data.jnt("ipad_freejoint").qpos[:3] = ipad[:3]
    raw_env._ipad_ori_pose = ipad.copy()


_STATE_RESTORERS = {
    "water_plant": _restore_water_plant,
    "water_plant_front_handle_oymotion": _restore_water_plant,
    "water_plant_inspire": _restore_water_plant,
    "water_plant_inspire_native_delta": _restore_water_plant,
    "water_plant_leap": _restore_water_plant,
    "water_plant_oymotion": _restore_water_plant,
    "water_plant_shadow_hand_e3m5": _restore_water_plant,
    "water_plant_sharpa_wave": _restore_water_plant,
    "water_plant_wuji": _restore_water_plant,
    "click_mouse": _restore_click_mouse,
    "click_mouse_inspire": _restore_click_mouse,
    "click_mouse_leap": _restore_click_mouse,
    "click_mouse_oymotion": _restore_click_mouse,
    "click_mouse_shadow_hand_e3m5": _restore_click_mouse,
    "click_mouse_sharpa_wave": _restore_click_mouse,
    "click_mouse_wuji": _restore_click_mouse,
    "pinch_tongs": _restore_pinch_tongs,
    "pinch_tongs_inspire": _restore_pinch_tongs,
    "pinch_tongs_leap": _restore_pinch_tongs,
    "pinch_tongs_oymotion": _restore_pinch_tongs_oymotion,
    "pinch_tongs_shadow_hand_e3m5": _restore_pinch_tongs,
    "pinch_tongs_sharpa_wave": _restore_pinch_tongs_sharpa_wave,
    "pinch_tongs_wuji": _restore_pinch_tongs_wuji,
    "fold_glasses": _restore_fold_glasses,
    "fold_glasses_inspire": _restore_fold_glasses,
    "fold_glasses_leap": _restore_fold_glasses,
    "fold_glasses_oymotion": _restore_fold_glasses,
    "fold_glasses_shadow_hand_e3m5": _restore_fold_glasses,
    "fold_glasses_sharpa_wave": _restore_fold_glasses,
    "fold_glasses_wuji": _restore_fold_glasses,
    "hammer_nail": _restore_hammer_nail,
    "hammer_nail_inspire": _restore_hammer_nail,
    "hammer_nail_inspire_native_delta": _restore_hammer_nail,
    "hammer_nail_leap": _restore_hammer_nail,
    "hammer_nail_oymotion": _restore_hammer_nail,
    "hammer_nail_shadow_hand_e3m5": _restore_hammer_nail,
    "hammer_nail_sharpa_wave": _restore_hammer_nail,
    "hammer_nail_wuji": _restore_hammer_nail,
    "pick_bucket": _restore_pick_bucket,
    "pick_bucket_inspire": _restore_pick_bucket,
    "pick_bucket_inspire_native_delta": _restore_pick_bucket,
    "pick_bucket_inspire_native_delta_galr_projected": _restore_pick_bucket,
    "pick_bucket_leap": _restore_pick_bucket,
    "pick_bucket_leap_native_delta": _restore_pick_bucket,
    "pick_bucket_leap_native_delta_galr_projected": _restore_pick_bucket,
    "pick_bucket_oymotion": _restore_pick_bucket,
    "pick_bucket_shadow_hand_e3m5": _restore_pick_bucket,
    "pick_bucket_sharpa_wave": _restore_pick_bucket,
    "pick_bucket_sharpa_wave_native_delta": _restore_pick_bucket,
    "pick_bucket_sharpa_wave_native_delta_galr_projected": _restore_pick_bucket,
    "pick_bucket_wuji": _restore_pick_bucket,
    "bimanual_assembly": _restore_bimanual_assembly,
    "bimanual_assembly_inspire": _restore_bimanual_assembly,
    "bimanual_assembly_leap": _restore_bimanual_assembly,
    "bimanual_assembly_oymotion": _restore_bimanual_assembly,
    "bimanual_assembly_sharpa_wave": _restore_bimanual_assembly,
    "bimanual_assembly_wuji": _restore_bimanual_assembly,
    "bimanual_hanoi": _restore_bimanual_hanoi,
    "bimanual_hanoi_inspire": _restore_bimanual_hanoi,
    "bimanual_hanoi_leap": _restore_bimanual_hanoi,
    "bimanual_hanoi_oymotion": _restore_bimanual_hanoi,
    "bimanual_hanoi_sharpa_wave": _restore_bimanual_hanoi,
    "bimanual_hanoi_wuji": _restore_bimanual_hanoi,
    "bimanual_microwave_cook": _restore_bimanual_microwave_cook,
    "bimanual_microwave_cook_inspire": _restore_bimanual_microwave_cook,
    "bimanual_microwave_cook_leap": _restore_bimanual_microwave_cook_leap,
    "bimanual_microwave_cook_oymotion": _restore_bimanual_microwave_cook,
    "bimanual_microwave_cook_sharpa_wave": _restore_bimanual_microwave_cook,
    "bimanual_microwave_cook_wuji": _restore_bimanual_microwave_cook,
    "bimanual_photograph": _restore_bimanual_photograph,
    "bimanual_photograph_inspire": _restore_bimanual_photograph,
    "bimanual_photograph_leap": _restore_bimanual_photograph,
    "bimanual_photograph_oymotion": _restore_bimanual_photograph,
    "bimanual_photograph_sharpa_wave": _restore_bimanual_photograph,
    "bimanual_photograph_wuji": _restore_bimanual_photograph,
    "bimanual_unlock_ipad": _restore_bimanual_unlock_ipad,
    "bimanual_unlock_ipad_inspire": _restore_bimanual_unlock_ipad,
    "bimanual_unlock_ipad_leap": _restore_bimanual_unlock_ipad,
    "bimanual_unlock_ipad_oymotion": _restore_bimanual_unlock_ipad,
    "bimanual_unlock_ipad_sharpa_wave": _restore_bimanual_unlock_ipad,
    "bimanual_unlock_ipad_wuji": _restore_bimanual_unlock_ipad,
}


def has_restorer(task_id):
    """True if `task_id` has a registered initial-state restorer."""
    return task_id in _STATE_RESTORERS


def restore_initial_state(env, task_id, config, state_vec):
    """Patch a freshly reset env to match the recorded `state[0]`.

    Args:
        env: a wrapped env (e.g. the one returned by `TaskConfig.get_environment`).
        task_id: key into the task registry; must satisfy `has_restorer(task_id)`.
        config: the matching `TaskConfig` (used for `proprio_keys`).
        state_vec: 1-D recorded proprio vector for `t = 0`.

    Returns:
        The refreshed wrapped observation produced after the restore + mj_forward.
    """
    '''
    1. 拿到原始 MuJoCo env
    2. 根据 config.proprio_keys 把扁平的 state_vec 拆回多个字段
    3. 根据 task_id 找到对应任务的恢复函数
    4. 修改 MuJoCo model/data 中的物体位姿
    5. 调用 mujoco.mj_forward 更新物理状态
    6. 返回恢复后的 observation

    env        包装后的环境，比如 TaskConfig.get_environment() 返回的 env
    task_id    任务名，例如 "bimanual_assembly"
    config     对应的 TaskConfig，用来拿 config.proprio_keys
    state_vec  demo 里记录的第一帧扁平 proprio 向量

    '''
    raw_env = env.unwrapped
    ref_state = raw_env._compute_observation()["state"]
    parts = _split_state_by_proprio(state_vec, config.proprio_keys, ref_state)
    _STATE_RESTORERS[task_id](raw_env, parts)
    mujoco.mj_forward(raw_env._model, raw_env._data)
    return env.observation(raw_env._compute_observation())
