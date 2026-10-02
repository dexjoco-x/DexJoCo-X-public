"""DexJoCo state/action adapters for UniDex's universal 82-D layout."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


MAPPED_JOINT_DIM = 32
ALLEGRO_FAAS_SLOTS = np.array([5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 1, 0, 2, 3], dtype=np.int64)
INSPIRE_FAAS_SLOTS = np.array([1, 2, 3, 4, 7, 8, 12, 13, 17, 18, 22, 23], dtype=np.int64)
# DexJoCo Sharpa Wave order:
#   thumb CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP,
#   index/middle/ring MCP_FE, MCP_AA, PIP, DIP,
#   pinky CMC, MCP_FE, MCP_AA, PIP, DIP.
#
# Sharpa's topology is the same as UniDex's 22-DoF Shadow hand.  In particular,
# slot 26 is the fifth thumb actuator (MCP ab/adduction) and slot 25 is the
# pinky metacarpal actuator; neither belongs in the generic 27--31 extension.
SHARPA_WAVE_FAAS_SLOTS = np.array(
    [1, 0, 3, 26, 4, 7, 6, 8, 9, 12, 11, 13, 14, 17, 16, 18, 19, 25, 22, 21, 23, 24],
    dtype=np.int64,
)
# Sharpa's FE/PIP/DIP coordinates are already positive for flexion.  Its CMC
# and long-finger AA axes need a sign flip to match the official FAAS motion
# direction.  The thumb MCP-AA axis is the exception: FK comparison against
# the official Shadow slot 26 shows that its native positive direction already
# matches FAAS.  Keep this per-axis calibration explicit instead of applying a
# blanket rule to every joint whose name contains ``AA``.
SHARPA_WAVE_FAAS_SCALE = np.array(
    [1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1],
    dtype=np.float32,
)
SHARPA_WAVE_FAAS_OFFSET = np.zeros(22, dtype=np.float32)
INSPIRE_ACTIVE_JOINT_DIM = 6
INSPIRE_FULL_JOINT_DIM = 12
INSPIRE_ACTIVE_TO_FULL_SLOTS = np.array([0, 1, 4, 6, 8, 10], dtype=np.int64)
UNIDEX_IDENTITY_POSE9 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

WRIST_FRAME_DEXJOCO_ATTACHMENT = "dexjoco_attachment"
WRIST_FRAME_UNIDEX_HAND_CANONICAL = "unidex_hand_canonical"
WRIST_FRAME_CONVENTIONS = {
    WRIST_FRAME_DEXJOCO_ATTACHMENT,
    WRIST_FRAME_UNIDEX_HAND_CANONICAL,
}

# UniDex/src/assets/utils/hand_utils.json::hand_transforms.Inspire.right.
# UniDex's pretraining datasets right-multiply the raw wrist transform by this
# fixed rotation before constructing states and relative action trajectories.
# It is rotation-only: the mesh-origin offsets in the DexJoCo XML are not TCP
# calibration and must not be folded into this transform.
UNIDEX_INSPIRE_RIGHT_WRIST_TRANSFORM = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_HAND_ALIASES = {
    "allegro": "Allegro",
    "allegro_right": "Allegro",
    "inspire": "Inspire",
    "inspire_right": "Inspire",
    "sharpa": "SharpaWave",
    "sharpawave": "SharpaWave",
    "sharpa-wave": "SharpaWave",
    "sharpa_wave": "SharpaWave",
    "sharpa_wave_left": "SharpaWave",
    "sharpa_wave_right": "SharpaWave",
}

DEXJOCO_HAND_SPECS = {
    "Allegro": {"action_dim": 23, "full_state_dim": 38, "action_rotvec_dim": 22},
    "Inspire": {"action_dim": 13, "full_state_dim": 28, "action_rotvec_dim": 12},
    "SharpaWave": {"action_dim": 29, "full_state_dim": 44, "action_rotvec_dim": 28},
}


def _as_float_array(value, dtype=np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def normalize_dexjoco_hand_type(hand_type: str) -> str:
    try:
        return _HAND_ALIASES[str(hand_type).strip().lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(DEXJOCO_HAND_SPECS))
        raise ValueError(f"Unsupported DexJoCo hand type {hand_type!r}; expected one of {supported}") from exc


def dexjoco_hand_spec(hand_type: str) -> dict[str, int]:
    return dict(DEXJOCO_HAND_SPECS[normalize_dexjoco_hand_type(hand_type)])


def normalize_wrist_frame_convention(convention: str) -> str:
    normalized = str(convention).strip().lower()
    aliases = {
        "attachment": WRIST_FRAME_DEXJOCO_ATTACHMENT,
        "raw": WRIST_FRAME_DEXJOCO_ATTACHMENT,
        "canonical": WRIST_FRAME_UNIDEX_HAND_CANONICAL,
        "unidex_canonical": WRIST_FRAME_UNIDEX_HAND_CANONICAL,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in WRIST_FRAME_CONVENTIONS:
        supported = ", ".join(sorted(WRIST_FRAME_CONVENTIONS))
        raise ValueError(
            f"Unsupported wrist-frame convention {convention!r}; expected one of {supported}"
        )
    return normalized


def unidex_hand_wrist_transform(
    hand_type: str,
    side: str = "right",
    wrist_frame_convention: str = WRIST_FRAME_DEXJOCO_ATTACHMENT,
) -> np.ndarray:
    """Return the fixed DexJoCo-attachment -> UniDex-wrist transform."""
    convention = normalize_wrist_frame_convention(wrist_frame_convention)
    if convention == WRIST_FRAME_DEXJOCO_ATTACHMENT:
        return np.eye(4, dtype=np.float32)
    canonical = normalize_dexjoco_hand_type(hand_type)
    if canonical == "Inspire" and side == "right":
        return UNIDEX_INSPIRE_RIGHT_WRIST_TRANSFORM.copy()
    raise ValueError(
        "The UniDex canonical wrist transform is currently calibrated only for "
        f"the Inspire right hand, got hand={canonical!r}, side={side!r}"
    )


def dexjoco_allegro16_to_faas32(q16: np.ndarray) -> np.ndarray:
    q16 = _as_float_array(q16)
    if q16.shape[-1] != 16:
        raise ValueError(f"Expected Allegro joints with last dim 16, got {q16.shape}")
    out = np.zeros(q16.shape[:-1] + (MAPPED_JOINT_DIM,), dtype=q16.dtype)
    out[..., ALLEGRO_FAAS_SLOTS] = q16
    return out


def faas32_to_dexjoco_allegro16(faas32: np.ndarray) -> np.ndarray:
    faas32 = _as_float_array(faas32)
    if faas32.shape[-1] != MAPPED_JOINT_DIM:
        raise ValueError(f"Expected FAAS joints with last dim {MAPPED_JOINT_DIM}, got {faas32.shape}")
    return faas32[..., ALLEGRO_FAAS_SLOTS]


def dexjoco_inspire_active6_to_full12(active6: np.ndarray) -> np.ndarray:
    active6 = _as_float_array(active6)
    if active6.shape[-1] != INSPIRE_ACTIVE_JOINT_DIM:
        raise ValueError(f"Expected Inspire active joints with last dim 6, got {active6.shape}")
    out = np.zeros(active6.shape[:-1] + (INSPIRE_FULL_JOINT_DIM,), dtype=active6.dtype)
    out[..., INSPIRE_ACTIVE_TO_FULL_SLOTS] = active6
    out[..., 2] = 1.334 * active6[..., 1]
    out[..., 3] = 0.667 * active6[..., 1]
    out[..., [5, 7, 9, 11]] = 1.06399 * active6[..., 2:6] - 0.04545
    return out


def dexjoco_inspire_full12_to_active6(full12: np.ndarray) -> np.ndarray:
    full12 = _as_float_array(full12)
    if full12.shape[-1] != INSPIRE_FULL_JOINT_DIM:
        raise ValueError(f"Expected Inspire full joints with last dim 12, got {full12.shape}")
    return full12[..., INSPIRE_ACTIVE_TO_FULL_SLOTS]


def dexjoco_inspire_full12_to_faas32(full12: np.ndarray) -> np.ndarray:
    full12 = _as_float_array(full12)
    if full12.shape[-1] != INSPIRE_FULL_JOINT_DIM:
        raise ValueError(f"Expected Inspire full joints with last dim 12, got {full12.shape}")
    out = np.zeros(full12.shape[:-1] + (MAPPED_JOINT_DIM,), dtype=full12.dtype)
    out[..., INSPIRE_FAAS_SLOTS] = full12
    return out


def faas32_to_dexjoco_inspire_full12(faas32: np.ndarray) -> np.ndarray:
    faas32 = _as_float_array(faas32)
    if faas32.shape[-1] != MAPPED_JOINT_DIM:
        raise ValueError(f"Expected FAAS joints with last dim {MAPPED_JOINT_DIM}, got {faas32.shape}")
    return faas32[..., INSPIRE_FAAS_SLOTS]


def dexjoco_sharpa_wave22_to_faas32(q22: np.ndarray, side: str = "right") -> np.ndarray:
    """Map native Sharpa Wave joints to UniDex FAAS, including sign alignment."""
    q22 = _as_float_array(q22)
    if q22.shape[-1] != 22:
        raise ValueError(f"Expected Sharpa Wave joints with last dim 22, got {q22.shape}")
    if side not in {"right", "left"}:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    aligned = q22 * SHARPA_WAVE_FAAS_SCALE.astype(q22.dtype) + SHARPA_WAVE_FAAS_OFFSET.astype(q22.dtype)
    out = np.zeros(q22.shape[:-1] + (MAPPED_JOINT_DIM,), dtype=q22.dtype)
    out[..., SHARPA_WAVE_FAAS_SLOTS] = aligned
    return out


def faas32_to_dexjoco_sharpa_wave22(faas32: np.ndarray, side: str = "right") -> np.ndarray:
    """Invert the Sharpa Wave FAAS map back to native simulator joint signs."""
    faas32 = _as_float_array(faas32)
    if faas32.shape[-1] != MAPPED_JOINT_DIM:
        raise ValueError(f"Expected FAAS joints with last dim {MAPPED_JOINT_DIM}, got {faas32.shape}")
    if side not in {"right", "left"}:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    aligned = faas32[..., SHARPA_WAVE_FAAS_SLOTS]
    return (aligned - SHARPA_WAVE_FAAS_OFFSET.astype(aligned.dtype)) / SHARPA_WAVE_FAAS_SCALE.astype(
        aligned.dtype
    )


def dexjoco_pose7_to_matrix(pose7_wxyz: np.ndarray) -> np.ndarray:
    pose7_wxyz = _as_float_array(pose7_wxyz)
    if pose7_wxyz.shape[-1] != 7:
        raise ValueError(f"Expected DexJoCo pose last dim 7, got {pose7_wxyz.shape}")
    quat_wxyz = pose7_wxyz[..., 3:7]
    quat_xyzw = np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    matrix = np.zeros(pose7_wxyz.shape[:-1] + (4, 4), dtype=pose7_wxyz.dtype)
    matrix[..., :3, :3] = rot
    matrix[..., :3, 3] = pose7_wxyz[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def matrix_to_dexjoco_pose7(matrix: np.ndarray) -> np.ndarray:
    matrix = _as_float_array(matrix)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected transform matrix shape (..., 4, 4), got {matrix.shape}")
    quat_xyzw = Rotation.from_matrix(matrix[..., :3, :3]).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1)
    return np.concatenate([matrix[..., :3, 3], quat_wxyz], axis=-1).astype(matrix.dtype)


def matrix_to_unidex_pose9(matrix: np.ndarray) -> np.ndarray:
    matrix = _as_float_array(matrix)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected transform matrix shape (..., 4, 4), got {matrix.shape}")
    rot6d = matrix[..., :2, :3].reshape(matrix.shape[:-2] + (6,))
    return np.concatenate([matrix[..., :3, 3], rot6d], axis=-1).astype(matrix.dtype)


def dexjoco_pose7_to_unidex_pose9(pose7_wxyz: np.ndarray) -> np.ndarray:
    return matrix_to_unidex_pose9(dexjoco_pose7_to_matrix(pose7_wxyz))


def _normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def unidex_pose9_to_matrix(pose9: np.ndarray) -> np.ndarray:
    pose9 = _as_float_array(pose9)
    if pose9.shape[-1] != 9:
        raise ValueError(f"Expected UniDex pose last dim 9, got {pose9.shape}")
    first = _normalize(pose9[..., 3:6].astype(np.float64))
    second_raw = pose9[..., 6:9].astype(np.float64)
    second = _normalize(second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first)
    third = np.cross(first, second, axis=-1)
    rot = np.stack((first, second, third), axis=-2)
    matrix = np.zeros(pose9.shape[:-1] + (4, 4), dtype=pose9.dtype)
    matrix[..., :3, :3] = rot
    matrix[..., :3, 3] = pose9[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def unidex_pose9_to_dexjoco_pose7(pose9: np.ndarray) -> np.ndarray:
    return matrix_to_dexjoco_pose7(unidex_pose9_to_matrix(pose9))


def dexjoco_pose7_world_to_camera(
    pose7_world: np.ndarray,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
) -> np.ndarray:
    """Express a world-frame pose in MuJoCo's camera frame (x right, y up, z back)."""
    world_pose = dexjoco_pose7_to_matrix(pose7_world)
    camera_position_world = _as_float_array(camera_position_world, dtype=world_pose.dtype)
    camera_rotation_world = _as_float_array(camera_rotation_world, dtype=world_pose.dtype)
    if camera_position_world.shape[-1] != 3 or camera_rotation_world.shape[-2:] != (3, 3):
        raise ValueError("Camera position/rotation must have shapes (..., 3) and (..., 3, 3)")
    rotation_camera_from_world = np.swapaxes(camera_rotation_world, -1, -2)
    camera_pose = np.zeros_like(world_pose)
    camera_pose[..., :3, :3] = np.matmul(rotation_camera_from_world, world_pose[..., :3, :3])
    delta = world_pose[..., :3, 3] - camera_position_world
    camera_pose[..., :3, 3] = np.matmul(rotation_camera_from_world, delta[..., None])[..., 0]
    camera_pose[..., 3, 3] = 1.0
    return matrix_to_dexjoco_pose7(camera_pose)


def compose_dexjoco_pose7_with_unidex_relative_pose9(
    current_pose7_world: np.ndarray,
    relative_pose9: np.ndarray,
) -> np.ndarray:
    current = dexjoco_pose7_to_matrix(current_pose7_world)
    relative = unidex_pose9_to_matrix(relative_pose9)
    return matrix_to_dexjoco_pose7(np.matmul(current, relative))


def _pack_action(pose9: np.ndarray, faas32: np.ndarray, side: str) -> np.ndarray:
    out = np.zeros(pose9.shape[:-1] + (82,), dtype=pose9.dtype)
    out[..., 0:9] = UNIDEX_IDENTITY_POSE9
    out[..., 9:18] = UNIDEX_IDENTITY_POSE9
    if side == "right":
        out[..., 0:9] = pose9
        out[..., 18:50] = faas32
    elif side == "left":
        out[..., 9:18] = pose9
        out[..., 50:82] = faas32
    else:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    return out


def _pack_bimanual_action(
    right_pose9: np.ndarray,
    left_pose9: np.ndarray,
    right_faas32: np.ndarray,
    left_faas32: np.ndarray,
) -> np.ndarray:
    return np.concatenate([right_pose9, left_pose9, right_faas32, left_faas32], axis=-1).astype(
        right_pose9.dtype
    )


def _unpack_action(action82: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    action82 = _as_float_array(action82)
    if action82.shape[-1] != 82:
        raise ValueError(f"Expected UniDex action last dim 82, got {action82.shape}")
    if side == "right":
        return action82[..., 0:9], action82[..., 18:50]
    if side == "left":
        return action82[..., 9:18], action82[..., 50:82]
    raise ValueError(f"side must be 'right' or 'left', got {side!r}")


def dexjoco_action23_to_unidex_action82(action23: np.ndarray, side: str = "right") -> np.ndarray:
    action23 = _as_float_array(action23)
    if action23.shape[-1] != 23:
        raise ValueError(f"Expected DexJoCo action last dim 23, got {action23.shape}")
    pose9 = dexjoco_pose7_to_unidex_pose9(action23[..., :7])
    return _pack_action(pose9, dexjoco_allegro16_to_faas32(action23[..., 7:23]), side)


def unidex_action82_to_dexjoco_action23(action82: np.ndarray, side: str = "right") -> np.ndarray:
    pose9, faas32 = _unpack_action(action82, side)
    return np.concatenate(
        [unidex_pose9_to_dexjoco_pose7(pose9), faas32_to_dexjoco_allegro16(faas32)],
        axis=-1,
    ).astype(np.asarray(action82).dtype)


def dexjoco_state38_to_unidex_state82(state38: np.ndarray, side: str = "right") -> np.ndarray:
    state38 = _as_float_array(state38)
    if state38.shape[-1] != 38:
        raise ValueError(f"Expected Hammer Nail full state last dim 38, got {state38.shape}")
    return dexjoco_action23_to_unidex_action82(state38[..., :23], side=side)


def dexjoco_inspire_action13_to_unidex_action82(action13: np.ndarray, side: str = "right") -> np.ndarray:
    action13 = _as_float_array(action13)
    if action13.shape[-1] != 13:
        raise ValueError(f"Expected DexJoCo Inspire action last dim 13, got {action13.shape}")
    pose9 = dexjoco_pose7_to_unidex_pose9(action13[..., :7])
    full12 = dexjoco_inspire_active6_to_full12(action13[..., 7:13])
    return _pack_action(pose9, dexjoco_inspire_full12_to_faas32(full12), side)


def unidex_action82_to_dexjoco_inspire_action13(action82: np.ndarray, side: str = "right") -> np.ndarray:
    pose9, faas32 = _unpack_action(action82, side)
    full12 = faas32_to_dexjoco_inspire_full12(faas32)
    active6 = dexjoco_inspire_full12_to_active6(full12)
    return np.concatenate([unidex_pose9_to_dexjoco_pose7(pose9), active6], axis=-1).astype(np.asarray(action82).dtype)


def dexjoco_inspire_state28_to_unidex_state82(state28: np.ndarray, side: str = "right") -> np.ndarray:
    state28 = _as_float_array(state28)
    if state28.shape[-1] != 28:
        raise ValueError(f"Expected Inspire Hammer Nail full state last dim 28, got {state28.shape}")
    return dexjoco_inspire_action13_to_unidex_action82(state28[..., :13], side=side)


def dexjoco_sharpa_wave_action29_to_unidex_action82(
    action29: np.ndarray,
    side: str = "right",
) -> np.ndarray:
    action29 = _as_float_array(action29)
    if action29.shape[-1] != 29:
        raise ValueError(f"Expected DexJoCo Sharpa Wave action last dim 29, got {action29.shape}")
    pose9 = dexjoco_pose7_to_unidex_pose9(action29[..., :7])
    faas32 = dexjoco_sharpa_wave22_to_faas32(action29[..., 7:29], side=side)
    return _pack_action(pose9, faas32, side)


def unidex_action82_to_dexjoco_sharpa_wave_action29(
    action82: np.ndarray,
    side: str = "right",
) -> np.ndarray:
    pose9, faas32 = _unpack_action(action82, side)
    q22 = faas32_to_dexjoco_sharpa_wave22(faas32, side=side)
    return np.concatenate([unidex_pose9_to_dexjoco_pose7(pose9), q22], axis=-1).astype(
        np.asarray(action82).dtype
    )


def dexjoco_sharpa_wave_state44_to_unidex_state82(
    state44: np.ndarray,
    side: str = "right",
) -> np.ndarray:
    state44 = _as_float_array(state44)
    if state44.shape[-1] != 44:
        raise ValueError(f"Expected Sharpa Wave Hammer Nail full state last dim 44, got {state44.shape}")
    return dexjoco_sharpa_wave_action29_to_unidex_action82(state44[..., :29], side=side)


def dexjoco_sharpa_wave_bimanual_action58_to_unidex_action82(action58: np.ndarray) -> np.ndarray:
    """Map recorded ``[right pose+q, left pose+q]`` Sharpa commands to UniDex."""
    action58 = _as_float_array(action58)
    if action58.shape[-1] != 58:
        raise ValueError(f"Expected bimanual Sharpa Wave action last dim 58, got {action58.shape}")
    right = action58[..., :29]
    left = action58[..., 29:58]
    return _pack_bimanual_action(
        dexjoco_pose7_to_unidex_pose9(right[..., :7]),
        dexjoco_pose7_to_unidex_pose9(left[..., :7]),
        dexjoco_sharpa_wave22_to_faas32(right[..., 7:29], side="right"),
        dexjoco_sharpa_wave22_to_faas32(left[..., 7:29], side="left"),
    )


def unidex_action82_to_dexjoco_sharpa_wave_bimanual_action58(action82: np.ndarray) -> np.ndarray:
    """Invert UniDex to the recorder's two contiguous 29-D arm blocks."""
    action82 = _as_float_array(action82)
    if action82.shape[-1] != 82:
        raise ValueError(f"Expected UniDex action last dim 82, got {action82.shape}")
    right = np.concatenate(
        [
            unidex_pose9_to_dexjoco_pose7(action82[..., 0:9]),
            faas32_to_dexjoco_sharpa_wave22(action82[..., 18:50], side="right"),
        ],
        axis=-1,
    )
    left = np.concatenate(
        [
            unidex_pose9_to_dexjoco_pose7(action82[..., 9:18]),
            faas32_to_dexjoco_sharpa_wave22(action82[..., 50:82], side="left"),
        ],
        axis=-1,
    )
    return np.concatenate([right, left], axis=-1).astype(np.asarray(action82).dtype)


def dexjoco_sharpa_wave_bimanual_state_to_unidex_state82(state: np.ndarray) -> np.ndarray:
    """Map ``[right pose, left pose, right q, left q, task state...]`` to UniDex."""
    state = _as_float_array(state)
    if state.shape[-1] < 58:
        raise ValueError(f"Expected bimanual Sharpa Wave state with at least 58 dims, got {state.shape}")
    return _pack_bimanual_action(
        dexjoco_pose7_to_unidex_pose9(state[..., 0:7]),
        dexjoco_pose7_to_unidex_pose9(state[..., 7:14]),
        dexjoco_sharpa_wave22_to_faas32(state[..., 14:36], side="right"),
        dexjoco_sharpa_wave22_to_faas32(state[..., 36:58], side="left"),
    )


def dexjoco_action_to_unidex_action82(action: np.ndarray, hand_type: str, side: str = "right") -> np.ndarray:
    canonical = normalize_dexjoco_hand_type(hand_type)
    if canonical == "Allegro":
        return dexjoco_action23_to_unidex_action82(action, side=side)
    if canonical == "Inspire":
        return dexjoco_inspire_action13_to_unidex_action82(action, side=side)
    return dexjoco_sharpa_wave_action29_to_unidex_action82(action, side=side)


def unidex_action82_to_dexjoco_action(action82: np.ndarray, hand_type: str, side: str = "right") -> np.ndarray:
    canonical = normalize_dexjoco_hand_type(hand_type)
    if canonical == "Allegro":
        return unidex_action82_to_dexjoco_action23(action82, side=side)
    if canonical == "Inspire":
        return unidex_action82_to_dexjoco_inspire_action13(action82, side=side)
    return unidex_action82_to_dexjoco_sharpa_wave_action29(action82, side=side)


def dexjoco_state_to_unidex_state82(state: np.ndarray, hand_type: str, side: str = "right") -> np.ndarray:
    canonical = normalize_dexjoco_hand_type(hand_type)
    if canonical == "Allegro":
        return dexjoco_state38_to_unidex_state82(state, side=side)
    if canonical == "Inspire":
        return dexjoco_inspire_state28_to_unidex_state82(state, side=side)
    return dexjoco_sharpa_wave_state44_to_unidex_state82(state, side=side)


def dexjoco_action_world_to_unidex_action82(
    action: np.ndarray,
    hand_type: str,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    side: str = "right",
    wrist_frame_convention: str = WRIST_FRAME_DEXJOCO_ATTACHMENT,
) -> np.ndarray:
    action = _as_float_array(action)
    camera_action = action.copy()
    camera_pose7 = dexjoco_pose7_world_to_camera(
        action[..., :7], camera_position_world, camera_rotation_world
    )
    camera_pose = dexjoco_pose7_to_matrix(camera_pose7)
    wrist_transform = unidex_hand_wrist_transform(
        hand_type,
        side=side,
        wrist_frame_convention=wrist_frame_convention,
    ).astype(camera_pose.dtype)
    camera_action[..., :7] = matrix_to_dexjoco_pose7(
        np.matmul(camera_pose, wrist_transform)
    )
    return dexjoco_action_to_unidex_action82(camera_action, hand_type, side=side)


def unidex_absolute_action82_to_dexjoco_action_world(
    action82: np.ndarray,
    hand_type: str,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    side: str = "right",
    wrist_frame_convention: str = WRIST_FRAME_DEXJOCO_ATTACHMENT,
) -> np.ndarray:
    """Invert an absolute UniDex policy-frame action into a world TCP target.

    ``dexjoco_action_world_to_unidex_action82`` first expresses the attachment
    pose in the fixed policy/camera frame and then right-multiplies the
    hand-specific UniDex wrist transform.  Absolute-action fine-tuning needs
    the exact inverse at deployment rather than re-anchoring on the live wrist.
    """
    action82 = _as_float_array(action82)
    camera_action = unidex_action82_to_dexjoco_action(
        action82,
        hand_type,
        side=side,
    )
    canonical_camera = dexjoco_pose7_to_matrix(camera_action[..., :7])
    wrist_transform = unidex_hand_wrist_transform(
        hand_type,
        side=side,
        wrist_frame_convention=wrist_frame_convention,
    ).astype(canonical_camera.dtype)
    attachment_camera = np.matmul(canonical_camera, np.linalg.inv(wrist_transform))

    camera_position_world = _as_float_array(
        camera_position_world,
        dtype=canonical_camera.dtype,
    )
    camera_rotation_world = _as_float_array(
        camera_rotation_world,
        dtype=canonical_camera.dtype,
    )
    if (
        camera_position_world.shape[-1] != 3
        or camera_rotation_world.shape[-2:] != (3, 3)
    ):
        raise ValueError(
            "Camera position/rotation must have shapes (..., 3) and (..., 3, 3)"
        )
    camera_world = np.zeros(
        camera_rotation_world.shape[:-2] + (4, 4),
        dtype=canonical_camera.dtype,
    )
    camera_world[..., :3, :3] = camera_rotation_world
    camera_world[..., :3, 3] = camera_position_world
    camera_world[..., 3, 3] = 1.0
    attachment_world = np.matmul(camera_world, attachment_camera)

    world_action = camera_action.copy()
    world_action[..., :7] = matrix_to_dexjoco_pose7(attachment_world)
    return world_action


def dexjoco_state_world_to_unidex_state82(
    state: np.ndarray,
    hand_type: str,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    side: str = "right",
    wrist_frame_convention: str = WRIST_FRAME_DEXJOCO_ATTACHMENT,
) -> np.ndarray:
    state = _as_float_array(state)
    camera_state = state.copy()
    camera_pose7 = dexjoco_pose7_world_to_camera(
        state[..., :7], camera_position_world, camera_rotation_world
    )
    camera_pose = dexjoco_pose7_to_matrix(camera_pose7)
    wrist_transform = unidex_hand_wrist_transform(
        hand_type,
        side=side,
        wrist_frame_convention=wrist_frame_convention,
    ).astype(camera_pose.dtype)
    camera_state[..., :7] = matrix_to_dexjoco_pose7(
        np.matmul(camera_pose, wrist_transform)
    )
    return dexjoco_state_to_unidex_state82(camera_state, hand_type, side=side)


def unidex_absolute_action82_to_relative(
    action82: np.ndarray,
    current_state82: np.ndarray,
    side: str = "right",
) -> np.ndarray:
    """Convert camera-frame absolute wrist targets to the official relative-action convention."""
    action82 = _as_float_array(action82)
    current_state82 = _as_float_array(current_state82, dtype=action82.dtype)
    action_pose9, _ = _unpack_action(action82, side)
    current_pose9, _ = _unpack_action(current_state82, side)
    current_matrix = unidex_pose9_to_matrix(current_pose9)
    target_matrix = unidex_pose9_to_matrix(action_pose9)
    relative_matrix = np.matmul(np.linalg.inv(current_matrix), target_matrix)
    relative = action82.copy()
    if side == "right":
        relative[..., 0:9] = matrix_to_unidex_pose9(relative_matrix)
        relative[..., 9:18] = UNIDEX_IDENTITY_POSE9
    elif side == "left":
        relative[..., 9:18] = matrix_to_unidex_pose9(relative_matrix)
        relative[..., 0:9] = UNIDEX_IDENTITY_POSE9
    else:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    return relative


def unidex_relative_action82_to_dexjoco_action(
    action82: np.ndarray,
    current_pose7_world: np.ndarray,
    hand_type: str,
    side: str = "right",
    wrist_frame_convention: str = WRIST_FRAME_DEXJOCO_ATTACHMENT,
) -> np.ndarray:
    relative_action = unidex_action82_to_dexjoco_action(action82, hand_type, side=side)
    relative_pose9, _ = _unpack_action(action82, side)
    current_attachment = dexjoco_pose7_to_matrix(current_pose7_world)
    relative_canonical = unidex_pose9_to_matrix(relative_pose9)
    wrist_transform = unidex_hand_wrist_transform(
        hand_type,
        side=side,
        wrist_frame_convention=wrist_frame_convention,
    ).astype(current_attachment.dtype)
    target_attachment = np.matmul(
        np.matmul(np.matmul(current_attachment, wrist_transform), relative_canonical),
        np.linalg.inv(wrist_transform),
    )
    out = relative_action.copy()
    out[..., :7] = matrix_to_dexjoco_pose7(target_attachment)
    return out
