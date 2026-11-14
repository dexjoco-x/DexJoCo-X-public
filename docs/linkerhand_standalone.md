# LinkerHand O6, L6, and L30

**English** | [中文](linkerhand_standalone.zh.md)

Three optional hand extensions support the same six task environments through `make_env`.
The default benchmark remains **seven hands, six tasks, and 2,100 Training v1 demonstrations**.
These three hands are outside that dataset and the published baseline averages.

## Run an environment

Install from the [main README](../README.md#get-started), then:

```python
from dexjoco.benchmark6.environments import make_env

env = make_env("hammer_nail", "linkerhand_l30", seed=154187, pixels=False)
try:
    observation, info = env.reset(seed=154187)
    observation, reward, terminated, truncated, info = env.step(env.hold_action())
finally:
    env.close()
```

Hand names: `linkerhand_o6`, `linkerhand_l6`, `linkerhand_l30`.
All six [task names](../README.md) are supported, including genuine left/right models for bimanual tasks.
Set `pixels=True` for image observations. An action contains six world-frame TCP increments followed by
native actuator targets; use `env.action_space` and `env.hand_metadata["sides"]` for the exact order and bounds.
The original seven-hand evaluation contracts and CLI continue to cover the fixed benchmark.

Assembled models are generated automatically in `$XDG_CACHE_HOME/dexjoco-x/`
(default `~/.cache/dexjoco-x/`). The cache is keyed by sources, adapter code, MuJoCo version,
and installation path. The shipped task XMLs are not overwritten.

## Replay the checked examples

The repository includes **18 small diagnostic command sequences**, one scene per hand–task pair.
They use the environment's ordinary actuator interface and physical contacts.
These are interaction checks, not trained-policy evaluation results or additional Training v1 data.

```bash
# All three hands and six tasks; no dataset download is required.
MUJOCO_GL=osmesa python examples/linkerhand/replay.py --workers 4

# One case, with a video.
MUJOCO_GL=osmesa python examples/linkerhand/replay.py \
  --hand linkerhand_l30 --task hammer_nail --video outputs/l30_hammer.mp4
```

The replay checks file hashes, task success and simulation warnings. L30 hammer replay also checks
that each scored strike contacts the nail head with the flat striking face.
Results are written to `outputs/linkerhand_replay.json`; the command exits nonzero on failure.
Use the numerical environment in the installation instructions for these saved-action checks.

## Simulation settings

Use the supplied MJCF with MuJoCo 3.4.0 and the 2 ms `implicitfast` timestep.
The position servos use `kp=5`, with `kv=1` for O6/L6 and `kv=1.05` for L30.
Passive damping and friction follow
`source.urdf`, with zero used for absent dynamics. The `0.001 kg*m^2` joint
armature regularizes small CAD inertias; it is a simulation setting, not a
measured motor inertia. The body masses, inertias, meshes, joint axes and
coupling ratios retain their source values.

L6/L30 actuator torque caps follow the source URDF effort values. L30's left
thumb has three actively driven joints with zero effort in the source URDF;
their MJCF caps use the corresponding right-hand limits (1 Nm). O6 retains
its tighter, original MJCF caps rather than the URDF's generic 100 Nm value.
These limits have not been calibrated on hardware. The URDF kinematics and
effort limits are retained; L6's MuJoCo extension is synchronized with the
MJCF coupling settings. The operational configuration is
`hand.xml` and the matching gains in `meta.yaml`.

Joint coupling uses a 4 ms constraint time constant. Contacts use the same
time constant, impedance `0.99`, and zero contact margin. This keeps
force-bearing contacts on the geometric surface, consistent with shared
task contact predicates. Original adjacent-link collision exclusions are preserved. L6 additionally
excludes palm-versus-thumb-metacarpal self-contact at its two-axis thumb mount:
the single palm convex hull fills a CAD clearance and incorrectly blocks
thumb opposition. A triangle-mesh check over 31×31 roll/pitch configurations
per side found no intersections (minimum clearance 0.181 mm). Both bodies
retain collision with objects and the other fingers. Hand colliders have priority 1, so their contact parameters apply
against ordinary priority-0 scene objects. This includes the friction parameters;
review this convention when composing scenes with other contact priorities.

## Checks

```bash
python -m unittest discover -s tests -p 'test_linkerhand*.py' -v
```

The tests cover provenance hashes, joint/actuator ordering, legal coupled
ranges, independent finger closure/release, bounded actuator torque, metadata
consistency, full-hand ramp/step/random commands, every active joint closing and reopening
with contacts enabled, and a rigid obstruction. A separate test verifies that
the L6 thumb root still collides with external objects.
In the obstruction fixture, the other active joints are held by constraints
so the tested finger must stop on contact and reopen. Contacts remain enabled;
no per-frame position resets or object attachments are used.

The contact suite checks all six left/right models in 30 scenarios. Its
acceptance bounds are 1 mm peak geometric overlap and 0.02 rad linkage error.
The task adapter adds a visible and collidable rigid flange bracket. It keeps the native wrist transform, hand mass and inertia; the bracket is a simulation fixture.
L30 includes a 230.5 mm forearm section before its wrist joint, which must be
included in the arm/TCP layout.

L30 servo damping reduces brief index/thumb impacts under full-range step
commands while retaining the same position gain and torque limits.

A task adapter may set `sides.left.palm_body_name` in hand metadata. iPad grasp grouping then distinguishes fingers below that palm, excluding the shared wrist/forearm. For L30, use `benchhand_left_native_lh_wrist_carpal_link0`. Models without this field keep the existing grouping. The additional regression is `python -m unittest discover -s tests -p test_ipad_hand_branches.py -v`.

## Sources

Original assets were contributed by [h228zhou](https://github.com/h228zhou) in [PR #4](https://github.com/dexjoco-x/DexJoCo-X-public/pull/4),
from [linker-bot/linker-sim-assets](https://github.com/linker-bot/linker-sim-assets/tree/d0224b8da58802fe7cb2e35dcc9771693c4543b6).
Per-hand `SOURCE_PROVENANCE.json` records original hashes and local corrections.
No explicit asset license was found in that pinned source; the L20 license does not cover these models.
See [third-party notices](../THIRD_PARTY_NOTICES.md).
