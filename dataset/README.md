# Dataset: download, read, and convert

**English** | [中文](README.zh.md)

**7 hands · 6 tasks · 2,100 training demonstrations · 946,156 frames · 20 Hz**

The [Hugging Face dataset](https://huggingface.co/datasets/darenrenjian/DexJoCo-X) provides six task archives. Each contains 50 demonstrations for each of the seven hands. The dataset version is `dataset-v1.0-corrected`.

## Download

After [installation](../README.md#get-started), run from the repository root:

```bash
export DATA_ROOT="$PWD/outputs/data"
python tools/dataset_packages.py download --tasks hammer_nail \
  --data-root "$DATA_ROOT"
```

Use `--tasks all` for the full dataset, or specify multiple task names. The tool extracts the archives and verifies SHA-256 checksums. The six archives total 8.242 GB; downloading and extracting them requires about 16.5 GB.

| Task | Demonstrations | Archive size (GB) |
|---|---:|---:|
| `hammer_nail` | 350 | 0.746 |
| `pick_bucket` | 350 | 0.962 |
| `single_hand_hanoi` | 350 | 1.334 |
| `bimanual_microwave_cook` | 350 | 2.081 |
| `bimanual_photograph` | 350 | 0.955 |
| `bimanual_unlock_ipad` | 350 | 2.164 |

Options: `--dry-run` previews the download, `--workers` sets parallelism, `--archives-root` sets the archive cache, and `--transport files` downloads individual raw files. The default archive cache is `<data-directory-name>_archives/task_packages/v1.0/`, next to the data directory.

For existing local task archives:

```bash
export PACK_ROOT=/path/to/task_packages
python tools/dataset_packages.py extract --tasks all \
  --index "$PACK_ROOT/PACKAGES.json" --archives-root "$PACK_ROOT" \
  --data-root "$DATA_ROOT"
```

The downloader pins revision `9f48e1404168227a22586af0cffb8371dbb7e1f1`. Dataset-card updates do not change the downloaded data.

## Directory layout and raw fields

```text
DATA_ROOT/bundle/
  manifests/<task>.json
  episodes/<task>/<hand>/<scene>/
    native.npz
    contract.json
    front.mp4
    wrist_right.mp4
    wrist_left.mp4       # bimanual tasks
```

The manifest records scenes, seeds, valid frame counts, and file hashes. `native.npz` contains `state`, `action`, state/action masks, `arm_mask`, `timestamp`, and `image_mask`.

Raw arrays order the right side before the left. Each side's state begins with seven wrist values: world position and a wxyz quaternion. Each side's action begins with six world-frame wrist translation and rotation increments, followed by hand actuator targets. Positions use meters and rotations use radians. Hand channel order and ranges are defined in each demonstration's `contract.json`. RGB videos are 256×256 at 20 Hz; use the valid frame count in the manifest.

## Read a demonstration

```bash
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
```

Use `examples.read_demo.read_demo` directly in Python. It converts wrist orientations to rotation vectors and returns 80-dimensional states, action chunks, masks, and RGB images in physical units. The right side occupies 0:40 and the left side 40:80. See the [quickstart](../docs/QUICKSTART.md).

## Convert to training formats

To create Parquet and video training views:

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT"
python dataset/tools/prepare_benchmark7_repr.py "$DATA_ROOT" --representation faas \
  --contracts representations/table1_sided_v3/codecs/faas7_project_v3_sided.json
```

The default control coordinates and FAAS views are written to `prepared/beingh05/` and `prepared/beingh05_faas/`. These directory names are compatibility keys. Both views use a general data format with 200-dimensional state/action arrays: right side 0:40, left side 40:80, and zero padding elsewhere. Wrist states use rotation vectors; FAAS uses 32 functional hand slots.

Conversion normalizes each hand–task pair using q01/q99, clips values to [-1, 1], and zeros invalid dimensions. These views differ from the reader's 80-dimensional physical values. Use matching normalization for training and inference. See the [policy example](../examples/pi05/README.md) for π₀.₅ statistics.

DexLatent conversion requires a compatible encoder and a matching binding record:

```bash
python dataset/tools/prepare_benchmark7_repr.py "$DATA_ROOT" --representation dexlatent \
  --contracts representations/table1_sided_v3/codecs/faas7_project_v3_sided.json \
  --checkpoint /path/to/selected.pt \
  --seal representations/table1_sided_v3/codecs/frozen_seal.json
```

`--seal` binds the encoder and control definitions. Output is written to `prepared/beingh05_dexlatent/`. Use a new working directory if a converted output already exists.

## Verification and related guides

```bash
python tools/dataset_packages.py verify --tasks hammer_nail --data-root "$DATA_ROOT"
python tools/dataset_smoke.py --data-root "$DATA_ROOT" --output ./outputs/data_check.json
```

For a partial download, select an installed task. See [version compatibility](../docs/VERSIONING.md), [camera settings](CAMERA_PROFILE.md) (Chinese), and [data licensing](LICENSE_SCOPE.md).
