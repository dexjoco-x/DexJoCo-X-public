# DexJoCo-X

[中文](README.md) · [Project website](https://darenrenjian.github.io/DexJoCo-X-website/) · [Paper PDF](https://darenrenjian.github.io/DexJoCo-X-website/assets/paper.pdf)

DexJoCo-X provides 2,100 demonstrations and a common simulation evaluation interface across seven dexterous hands and six tasks. Each of the 42 hand–task pairs contains 50 training demonstrations. Evaluation uses independent scene seeds and three independent runs of 50 rollouts per pair.

**Paper:** DexJoCo-X: Benchmarking Action Representations for Multi-Hand Dexterous Manipulation. **Authors:** Xiangwei Jiang, Yao Mu, Lixin Duan, Wen Li. Submitted to arXiv on October 2, 2026; the public identifier is pending. Citation metadata is available in [CITATION.cff](CITATION.cff).

The code and dataset repositories are currently private. This release contains the simulator and required assets, dataset download and conversion tools, Native/FAAS interfaces, and a π₀.₅ inference example. It does not include model training implementations, model or codec weights, or glove mappings. DexLatent evaluation requires a compatible codec supplied by the user.

## Install

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -c requirements/constraints-python311.txt \
  -e './simulator[benchmark,dataset,download]'
python tools/runtime_assets.py check
```

On a headless Ubuntu/Debian machine, install `libosmesa6`, then check all 42 environment combinations:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
python tools/portability_smoke.py
```

## Download and inspect demonstrations

The six task archives total 8.242 GB. Each task archive contains 350 demonstrations. Access to the current private Hugging Face dataset is required.

```bash
hf auth login
export DATA_ROOT="$PWD/outputs/data"
python tools/dataset_packages.py download --tasks hammer_nail \
  --data-root "$DATA_ROOT" --report ./outputs/download.json
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
```

The downloader pins the dataset revision and verifies archive and file SHA-256 hashes. Use `--tasks all` to download all six tasks. XHand microwave demonstrations include the corrected left-hand installation.

```python
from examples.read_demo import read_demo

sample = read_demo(data_root, task="hammer_nail", hand="allegro", horizon=25)
state = sample["state"]                # (80,), physical units
actions = sample["actions"]            # (25, 80), physical units
action_mask = sample["action_mask"]
rgb = sample["images"]["front"]         # uint8 RGB, (256, 256, 3)
```

The `arm40_v1` layout stores the right side in slots 0:40 and the left side in slots 40:80. Each side starts with six wrist channels followed by native hand actuator channels. Wrist states use world-frame positions and rotation vectors; actions use world-frame wrist increments and actuator targets. Units, masks, channel order, and camera names are documented in [the data interface](dataset/README.md).

## Connect your own policy

The [π₀.₅ example](examples/pi05/README.md) loads your own checkpoint, tokenizer, and training statistics through a fixed external RLinf version. It supports a 32-dimensional single-right-arm head and Ego-π left/right interleaved tokens. For interleaved inference, 50 tokens correspond to 25 physical time steps. Checkpoint format and normalization must match the inference adapter.

Start the policy server following that guide, then run a short integration check:

```bash
python evaluation/evaluate.py --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand allegro --task hammer_nail --episodes 1 \
  --action-horizon 25 --execution-horizon 8 \
  --diagnostic --max-steps 16 --output ./outputs/pi05_diagnostic
```

The checkpoint must support the selected hand and task. For formal evaluation, remove `--diagnostic --max-steps 16` and use `--episodes 150`; a six-task checkpoint can use `--task all`. See [the evaluation protocol](evaluation/README.md).

Raw physical tensors can be adapted to an external training framework. The optional Parquet conversion uses 200-dimensional normalized fields and is distinct from the 80-dimensional physical tensors; see [the download and conversion guide](dataset/DOWNLOAD.md). This repository provides no training implementation.

## Hands and tasks

Hands: Allegro, Inspire, LEAP, LinkerHand, Sharpa Wave, Wuji, and XHand. The historical internal key `linkerhand_l20` is retained for compatibility; its source geometry is from the official L25 model.

Single-arm tasks: `hammer_nail`, `pick_bucket`, `single_hand_hanoi`.

Bimanual tasks: `bimanual_microwave_cook`, `bimanual_photograph`, `bimanual_unlock_ipad`.

## Validation and licenses

Independent installation, task download, data conversion, all 42 environment combinations, and a real checkpoint integration run are documented in [the validation record](docs/DATASET_EVAL_RELEASE_VALIDATION.md). The real checkpoint run verifies the inference pipeline; it is not a new benchmark success-rate measurement.

Original project code is covered by [MIT](LICENSE), within [the stated scope](LICENSE_SCOPE.md). Third-party code and model assets retain their own terms; see [the notices](THIRD_PARTY_NOTICES.md) and [license status](LICENSE_STATUS.md). The final license for original demonstration data remains to be selected separately.
