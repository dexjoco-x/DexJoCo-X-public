"""Resolve release cells or explicitly supported frozen donor-source IDs."""
def get_config_mapping(hand="all", include_oymotion_variants=False):
    from dexjoco.benchmark6 import HANDS
    from .mappings import CONFIG_MAPPING
    if include_oymotion_variants or hand not in ("all", *HANDS):
        raise ValueError(f"Unsupported release hand: {hand!r}")
    return CONFIG_MAPPING if hand == "all" else {k: v for k, v in CONFIG_MAPPING.items() if v.hand == hand}

def get_task_config_class(task_name):
    from .mappings import CONFIG_MAPPING
    if task_name in CONFIG_MAPPING:
        return CONFIG_MAPPING[task_name]
    # Keep donor actions on their original controllers; never alias them to a
    # release environment with different delta limits or observations.
    from .legacy_sources import SOURCE_CONFIG_MAPPING
    if task_name in SOURCE_CONFIG_MAPPING:
        return SOURCE_CONFIG_MAPPING[task_name]
    raise KeyError(f"Unsupported task {task_name!r}; use task/hand from the release registry")

def make_task_config(task_name):
    return get_task_config_class(task_name)()
