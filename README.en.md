# DexJoCo-X

A multi-hand manipulation benchmark for comparing action representations across dexterous hands.

[Website](https://darenrenjian.github.io/DexJoCo-X-website/) · [Paper](https://arxiv.org/abs/2610.03278) · [Dataset](https://huggingface.co/datasets/darenrenjian/DexJoCo-X) · [中文](README.md)

![DexJoCo-X overview](https://darenrenjian.github.io/DexJoCo-X-website/assets/overview.webp)

**7 hands · 6 tasks · 42 hand–task pairs · 2,100 demonstrations**

DexJoCo-X brings different dexterous hands into a common set of tasks, data formats, and evaluation interfaces to compare Native, FAAS, and DexLatent action representations. Each hand–task pair contains 50 training demonstrations. Closed-loop evaluation uses independent scene seeds.

## What's included

- **Dataset**: 20 Hz demonstrations with RGB video, robot states, actions, valid-dimension masks, and task contracts.
- **Simulation environments**: robot models, objects, meshes, and textures for all seven hands and six tasks, bundled with the repository.
- **Evaluation tools**: a common policy interface, success criteria, and result logging, plus a π₀.₅ / Ego-π inference example and scene expansion tools for existing demonstrations.

Use your own training framework and policy checkpoints. DexLatent also requires a compatible local codec. Glove mappings and recording tools are not included.

**Hands**: Allegro, Inspire, LEAP, LinkerHand, Sharpa Wave, Wuji, and XHand.

| Type | Task | Configuration name |
|---|---|---|
| Single-arm | Hammer a nail | `hammer_nail` |
| Single-arm | Pick up a bucket | `pick_bucket` |
| Single-arm | Tower of Hanoi | `single_hand_hanoi` |
| Bimanual | Operate a microwave | `bimanual_microwave_cook` |
| Bimanual | Take a photograph | `bimanual_photograph` |
| Bimanual | Unlock a tablet | `bimanual_unlock_ipad` |

## Install

Use Python 3.11:

```bash
git clone https://github.com/darenrenjian/DexJoCo-X-public.git
cd DexJoCo-X-public
python3.11 -m venv .venv
source .venv/bin/activate
pip install -c requirements/constraints-python311.txt \
  -e './simulator[benchmark,dataset,download]'
python tools/runtime_assets.py check
```

For headless Ubuntu / Debian, use OSMesa:

```bash
sudo apt-get install libosmesa6
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
python tools/portability_smoke.py
```

The environment check covers all 42 hand–task pairs. See the [quickstart](docs/QUICKSTART.md) for the full workflow.

## Download and inspect demonstrations

The code and dataset are currently private; downloading requires an authorized Hugging Face account. The six task archives total approximately **8.24 GB**, with 50 demonstrations per hand in each archive. Downloading and extracting the full dataset requires approximately 16.5 GB of disk space.

Download the hammer task, verify its files, and inspect one Allegro demonstration:

```bash
export DATA_ROOT="$PWD/outputs/data"
python tools/dataset_packages.py download --tasks hammer_nail \
  --data-root "$DATA_ROOT"
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
```

Use `--tasks all` to download all tasks. The reader returns 80-dimensional states and action chunks in physical units, masks, and RGB images. Optional conversion produces Parquet training views. Field definitions, normalization, and conversion are documented in the [data interface](dataset/README.md) and [download guide](dataset/DOWNLOAD.md).

## Connect a policy and evaluate

A policy receives images, robot states, masks, and task instructions and returns action chunks. Start your policy server using the [π₀.₅ / Ego-π example](examples/pi05/README.md), then evaluate a supported hand and task:

```bash
python evaluation/evaluate.py --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand allegro --task hammer_nail --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/allegro_hammer
```

Each hand–task pair uses **three independent evaluations of 50 rollouts each**. Other policies can use the same interface. Action layout, representation, and prediction horizon must match the policy server; see the [evaluation protocol](evaluation/README.md).

## Documentation

| Topic | Guide |
|---|---|
| Installation through policy integration | [Quickstart](docs/QUICKSTART.md) |
| Data fields and conversion | [Data interface](dataset/README.md) · [Download guide](dataset/DOWNLOAD.md) |
| Simulation and evaluation | [Environments](simulator/README.md) · [Evaluation interface](evaluation/README.md) |
| π₀.₅ / Ego-π inference | [Integration example](examples/pi05/README.md) |
| Automated scene expansion | [Collection tools](collection/README.md) |
| Data and model versions | [Versioning](docs/VERSIONING.md) |

## Citation and licenses

Citation metadata is available in [BibTeX](CITATION.bib) and [CITATION.cff](CITATION.cff).

Original code uses [MIT](LICENSE) within the [documented scope](LICENSE_SCOPE.md). Original dataset contributions use **CC BY-NC-SA 4.0** within the [data license scope](dataset/LICENSE_SCOPE.md). Third-party code and assets retain their separate terms; see the [third-party notices](THIRD_PARTY_NOTICES.md) and [license status](LICENSE_STATUS.md) for sources, licenses, and unconfirmed scopes.
