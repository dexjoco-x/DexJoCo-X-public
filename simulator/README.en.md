# Simulation environments

[中文](README.md) | **English**

MuJoCo environments cover seven dexterous hands and six tasks. Robot models, objects, and textures are included in the package.

## Create an environment

After [installation](../README.en.md#get-started), run in Python:

```python
from dexjoco.benchmark6.environments import make_env

env = make_env("pick_bucket", "allegro", seed=0, pixels=False)
try:
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(env.hold_action())
finally:
    env.close()
```

`make_env(task, hand, seed=...)` accepts the tasks and hands listed on the homepage. Set `pixels=True` to enable image observations. `hold_action()` returns an action that holds the current pose for an environment check.

## Control and observations

Wrist actions are world-frame translation and rotation increments; hand actions are actuator targets. Actuator names, ranges, and success conditions are recorded in the [task contracts](../evaluation/contracts).

The common policy interface uses images, states, masks, and task instructions. See the [evaluation guide](../evaluation/README.en.md) and [camera settings](../dataset/CAMERA_PROFILE.md) (Chinese).

## Modules

- `dexjoco.benchmark6.environments`: environment entry point.
- `dexjoco.benchmark6.success_policy`: success predicates.
- `dexjoco.benchmark6.dataset`: demonstration reader.
- `dexjoco.autocollect`: scene expansion from existing demonstrations.

`teleoperation/` contains model assets referenced by the environments. See [third-party notices](../THIRD_PARTY_NOTICES.md) for sources and terms, and [version compatibility](../docs/VERSIONING.en.md) for identifiers.
