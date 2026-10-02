from __future__ import annotations

import numpy as np


ALLEGRO_RIGHT_HAND_TYPE = "allegro_right"

ALLEGRO_RIGHT_JOINT_NAMES = (
    "ffj0",
    "ffj1",
    "ffj2",
    "ffj3",
    "mfj0",
    "mfj1",
    "mfj2",
    "mfj3",
    "rfj0",
    "rfj1",
    "rfj2",
    "rfj3",
    "thj0",
    "thj1",
    "thj2",
    "thj3",
)

ALLEGRO_RIGHT_ACTUATOR_NAMES = (
    "ffa0",
    "ffa1",
    "ffa2",
    "ffa3",
    "mfa0",
    "mfa1",
    "mfa2",
    "mfa3",
    "rfa0",
    "rfa1",
    "rfa2",
    "rfa3",
    "tha0",
    "tha1",
    "tha2",
    "tha3",
)

ALLEGRO_RIGHT_HOME = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0),
    dtype=np.float32,
)

ALLEGRO_RIGHT_BIMANUAL_JOINT_NAMES = tuple(
    f"{name}_right" for name in ALLEGRO_RIGHT_JOINT_NAMES
)
ALLEGRO_RIGHT_BIMANUAL_ACTUATOR_NAMES = tuple(
    f"{name}_right" for name in ALLEGRO_RIGHT_ACTUATOR_NAMES
)
ALLEGRO_LEFT_BIMANUAL_JOINT_NAMES = (
    "rfj0_left",
    "rfj1_left",
    "rfj2_left",
    "rfj3_left",
    "mfj0_left",
    "mfj1_left",
    "mfj2_left",
    "mfj3_left",
    "ffj0_left",
    "ffj1_left",
    "ffj2_left",
    "ffj3_left",
    "thj0_left",
    "thj1_left",
    "thj2_left",
    "thj3_left",
)
ALLEGRO_LEFT_BIMANUAL_ACTUATOR_NAMES = (
    "rfa0_left",
    "rfa1_left",
    "rfa2_left",
    "rfa3_left",
    "mfa0_left",
    "mfa1_left",
    "mfa2_left",
    "mfa3_left",
    "ffa0_left",
    "ffa1_left",
    "ffa2_left",
    "ffa3_left",
    "tha0_left",
    "tha1_left",
    "tha2_left",
    "tha3_left",
)


def shadow_hand_e3m5_joint_targets_to_actuator_ctrl(*args, **kwargs):
    raise ValueError("Shadow is not a DexJoCo-X release embodiment")


def get_hammer_nail_hand_spec(hand_type: str):
    from .hand_specs import get_hammer_nail_hand_spec as impl

    return impl(hand_type)


def get_pick_bucket_hand_spec(hand_type: str):
    from .hand_specs import get_pick_bucket_hand_spec as impl

    return impl(hand_type)
