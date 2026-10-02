"""Additive cross-hand benchmark; the module name is kept for CLI compatibility."""

VERSION = "benchmark7_v3"
ARTIFACT_ROOT = "artifacts/benchmark7/v3"
HANDS = ("allegro", "leap", "xhand", "inspire", "wuji", "sharpa_wave", "linkerhand_l20")
# User-approved main selection. These are candidates, NOT a success guarantee.
TARGET_TASKS = (
    "pick_bucket", "hammer_nail", "single_hand_hanoi",
    "bimanual_microwave_cook", "bimanual_unlock_ipad", "bimanual_photograph",
)
CANDIDATE_TASKS = TARGET_TASKS
