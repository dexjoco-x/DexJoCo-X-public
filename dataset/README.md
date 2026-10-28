# Dataset

**English** | [中文](README.zh.md)

**Training v1 is the current benchmark dataset for both training and demonstration replay.**

7 hands · 6 tasks · 2,100 training demonstrations · 933,865 frames · 20 Hz.
Every hand–task pair has 50 demonstrations. Float64 commands, LeRobot and Being-H
views describe the same trajectories and use identical videos.

## Download

Complete the [installation](../README.md#get-started), then run:

```bash
export DATA_ROOT="$PWD/outputs/training_v1"
python tools/dataset_packages.py download --tasks all --data-root "$DATA_ROOT"
```

The tool downloads one 5.21 GB archive, verifies its SHA-256, extracts it and checks
every file. Allow 19 GB of free space for the archive and extracted files. All six
tasks and both training readers are included, even if an older command specifies
one task with `--tasks`. `--dry-run` displays the download before starting it.

The default cache is `training_v1_archives/training_v1/` beside the data directory;
use `--archives-root` to change it. The revision and hashes are fixed in
[`PACKAGES.json`](PACKAGES.json). A different existing dataset is never overwritten.

For an already downloaded Training v1 archive:

```bash
python tools/dataset_packages.py extract --archives-root /path/to/training_v1 \
  --data-root "$DATA_ROOT"
```

## Choose a reader

| Use | Location in the extracted package |
|---|---|
| Demonstration replay and physical-unit arrays | `bundle/` |
| Official LeRobot v3 (tested with 0.4.4) | `prepared/lerobot_final_v3/` |
| Being-H reader | `prepared/beingh_final_v3/` |
| Matching normalization/decoder statistics | `prepared/*_train_stats.json` |

These are formats of one dataset, not alternative dataset versions. No conversion
is needed to use either prepared reader. Float32 labels are already normalized;
decode them using the included statistics and `Native200PolicyAdapter`, retaining
float64 commands through the simulator interface. See [training usage](../docs/TRAINING_CONVERSION.md).

## Files and units

```text
bundle/
  catalog.json
  manifests/<task>.json
  episodes/<task>/<hand>/<scene>/
    native.npz
    contract.json
    front.mp4
    wrist_right.mp4
    wrist_left.mp4       # bimanual tasks
prepared/
  lerobot_final_v3/train/<task>/<hand>/
  beingh_final_v3/train/<task>/<hand>/
metadata/VERIFICATION.json
SHA256SUMS
```

The manifest records scene seeds, valid frame counts and hashes. `native.npz`
contains states, float64 physical actions, exact float32 training labels, masks
and timestamps. RGB videos are 256×256 at 20 Hz. Observations precede each action.

Raw arrays order the right side before the left. Each side's state begins with
world position and a wxyz quaternion; each action begins with six world-frame
wrist translation/rotation increments, followed by hand actuator targets. Positions
use meters and rotations use radians. Hand channels and ranges are in `contract.json`.
Raw camera masks use `[front, wrist_right, wrist_left]`; the policy uses
`[base, wrist_left, wrist_right]`, where `base` means `front`. Match cameras by name.

```bash
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
python tools/dataset_packages.py verify --data-root "$DATA_ROOT"
```

The example reader returns 80-dimensional physical states/actions, masks and RGB.
See [replay checks](../docs/REPLAY_REPAIR.md), [evaluation](../evaluation/README.md)
and [licensing](LICENSE_SCOPE.md). Historical releases are documented only in
[version history](../docs/VERSIONING.md); use Training v1 for new comparisons.
