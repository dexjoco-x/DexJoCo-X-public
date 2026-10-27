# Versions and compatibility

**English** | [中文](VERSIONING.zh.md)

| Version | Contents |
|---|---|
| `dataset-v1.1-native-replay` | 2,100 replay-verified demonstrations; 939,356 synchronized frames |
| `runtime-v10` | Paired models and a pinned numerical environment for v1.1 |
| `dataset-v1.0-corrected` / `runtime-v9` | Previous public snapshot, retained for reproducibility |
| `paper-v1` | Earlier training snapshots, checkpoints and reported policy results |

The v1.1 update corrects XHand left mounts in the photograph and tablet environments,
recollects 220 physical action sequences, removes unexecuted terminal suffixes from
918 other sequences, and exports synchronized observations for every demonstration.
The original 50-demonstration budget per hand–task pair and task success rules remain.

Use the [reference environment](REPLAY_REPAIR.md) with v1.1. The paper checkpoints
have not been retrained for this update; identify the data/runtime version when
reporting training or policy evaluation results. Regenerate converted training views
from the selected raw dataset version.

The earlier raw snapshot is available at dataset revision
`9f48e1404168227a22586af0cffb8371dbb7e1f1`.

`linkerhand_l20` is a compatibility key in files and contracts. Its source left/right
URDFs were modified from the official L25 files. Documentation uses `LinkerHand`;
the key and model geometry are preserved.
