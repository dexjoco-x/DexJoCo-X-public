# Current benchmark and version history

**English** | [中文](VERSIONING.zh.md)

## Current standard

Use **Training v1** (`training-float32-v1`) with the current code and
[reference environment](REPLAY_REPAIR.md). All download commands in the current
guides resolve to this one dataset: 2,100 demonstrations, 933,865 frames, seven
hands and six tasks. Its float64, LeRobot and Being-H files represent the same
executions. [Download](../dataset/README.md).

Training uses 50 demonstrations per hand–task pair. Policy evaluation uses three
independent runs of 50 rollouts per pair, with the fixed scene seeds and success
criteria in the [evaluation guide](../evaluation/README.md). Report the dataset
identifier, code commit, model checkpoint and evaluation settings alongside results.

Training v1 uses the existing reference physics and corrected runtime models.
Its float32 labels use the bound `quantile_affine_v3` statistics and float64 decoder.
Updating the default download does not change its archive, trajectories or task rules.

## Historical snapshots

Earlier snapshots are archived for tracing prior results. They are not alternative
choices for new benchmark runs and are no longer in the default data directory.

| Archived snapshot | Purpose |
|---|---|
| [Raw v1.1](https://huggingface.co/datasets/darenrenjian/DexJoCo-X/tree/dataset-v1.1-native-replay) | Earlier float64 release and replay investigation |
| [Initial public snapshot](https://huggingface.co/datasets/darenrenjian/DexJoCo-X/tree/dataset-v1.0-corrected) | Reproduce earlier public downloads |
| `paper-v1` | Earlier paper experiments and model checkpoints |

The paper's reported results used earlier data; they have not been relabeled as
Training v1 results. Earlier checkpoints keep their own statistics and decoder.
Training v1 and Raw v1.1 share scene IDs, so combining them does not add independent
scenes. The [issue history](https://github.com/dexjoco-x/DexJoCo-X-public/issues/1)
records the corrections.

`linkerhand_l20` is a retained file/contract key. Its source geometry is based on
the official L25 model; public descriptions use LinkerHand.
