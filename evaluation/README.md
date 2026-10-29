# Policy integration and evaluation

**English** | [中文](README.zh.md)

The current benchmark uses **Training v1**, the [reference environment](../docs/REPLAY_REPAIR.md), and the evaluation protocol below. Float64 and LeRobot are formats of the same trajectories, not different evaluation standards. Record dataset version `training-float32-v1` and the code commit with your results.

The evaluator connects to a policy server, sends images, robot states, masks, and task instructions, and executes the returned action chunks in simulation.

## Run an evaluation

Start the [π₀.₅ / Ego-π policy server](../examples/pi05/README.md), then run:

```bash
python evaluation/evaluate.py --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand allegro --task hammer_nail --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/allegro_hammer
```

Use `--task all` to evaluate all six tasks for the selected hand, and `--video` to save recordings. Each hand–task pair is evaluated in three independent runs of 50 rollouts. Results record per-episode success, repeat indices, scene and policy seeds, checkpoint details, and protocol information.

Formal runs require `--episodes 150`; smaller counts require `--diagnostic`. Outputs automatically record the Git commit and working-tree status, evaluation source hashes, and benchmark dataset version/archive hash. Checkpoint training-data identity is recorded separately in server metadata when supplied by the policy.

The environment judges success using geometric, contact, and temporal conditions. Policy observations exclude object ground-truth states. Scene seeds are `10_000_000 + task_index × 1000 + episode_index`, where `episode_index` ranges from 0 to 149; policy seeds range from 17 to 166.

## States and actions

The policy server returns denormalized `H × 80` action chunks in physical units. The evaluator executes the first eight steps before requesting another chunk. Match `--action-horizon` to the server's prediction horizon. Ego-π's 50 alternating tokens represent 25 physical time steps.

| Layout | Action | State |
|---|---|---|
| `arm40_v1` | Right 0:40, left 40:80; per side: wrist 0:6, hand 6:38, reserved 38:40 | Per side: wrist position/rotvec 0:6, hand 6:38, reserved 38:40 |
| `grouped80_v1` | Right arm 0:6, left arm 6:12, right hand 12:44, left hand 44:76, reserved 76:80 | Right wrist position/quaternion 0:7, left wrist 7:14, right hand 14:46, left hand 46:78, reserved 78:80 |

Wrist positions use meters and rotations use radians. Single-arm tasks mask out the absent side. Hand channel order and ranges are defined in the [task contracts](contracts).

`--representation` supports three hand action representations: `native` for each hand's control coordinates, `faas` for function-aligned slots, and `dexlatent` for learned latents. Match the layout, representation, and normalization statistics to the policy. DexLatent also requires `./simulator[dexlatent]` and a matching encoder supplied through `BENCHMARK7_DEXLATENT_CHECKPOINT`.

## Connect another policy

Implement `infer(observation, noise=...)` to return a finite, denormalized `H × 80` array. The loader `my_policy.load(checkpoint_path)` returns this policy object:

```bash
python evaluation/serve80.py --factory my_policy:load \
  --checkpoint /path/to/checkpoint \
  --layout arm40_v1 --representation native --action-horizon 30
```

The adapter converts the model's action format to the evaluation interface. The model's own action head need not be 80-dimensional.

## Short diagnostic run

Use `--episodes 1 --diagnostic --max-steps 16` to check the policy–simulator connection. Diagnostic output is marked `diagnostic_only: true` and recorded separately from full success-rate evaluations.

Environment versions and model hashes are recorded in `models.json`. See [version compatibility](../docs/VERSIONING.md).

Formal evaluation checks the reference numerical environment and saves numerical_environment.json with package/build fingerprints and BLAS settings. A mismatch requires the reference installation or an explicitly diagnostic run.
