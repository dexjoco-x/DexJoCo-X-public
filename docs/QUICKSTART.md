# Quickstart

**English** | [中文](QUICKSTART.zh.md)

Run all commands from the repository root. First complete the [installation steps](../README.md#get-started).

## 1. Check the environments

```bash
python tools/portability_smoke.py
```

This checks all 42 environments across seven hands and six tasks. On a headless Ubuntu / Debian machine, use OSMesa:

```bash
sudo apt-get install libosmesa6
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
python tools/portability_smoke.py
```

## 2. Download and inspect a demonstration

```bash
export DATA_ROOT="$PWD/outputs/training_v1"
python tools/dataset_packages.py download --tasks all \
  --data-root "$DATA_ROOT"
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
```

Training v1 contains all six tasks, with 50 demonstrations per hand–task pair, plus matching LeRobot and Being-H views. The reader returns states, action chunks, and masks, and saves the selected RGB frames to a new output directory.

You can also read a demonstration in Python:

```python
from examples.read_demo import read_demo

sample = read_demo(data_root, task="hammer_nail", hand="allegro", horizon=25)
state = sample["state"]           # (80,), physical units
actions = sample["actions"]      # (25, 80), physical units
rgb = sample["images"]["front"]   # uint8 RGB, (256, 256, 3)
```

The default download is the complete 5.21 GB Training v1 package. See the [dataset guide](../dataset/README.md) for fields and format conversion.

## 3. Connect a policy and evaluate

Start a policy server using the [π₀.₅ / Ego-π example](../examples/pi05/README.md), then run:

```bash
python evaluation/evaluate.py --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand allegro --task hammer_nail --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/allegro_hammer
```

This example requires a policy supporting Allegro hammering. Each action chunk predicts 25 steps and executes the first eight. Evaluation consists of three independent runs of 50 rollouts each. Match the action layout and prediction horizon to the policy server. See the [evaluation guide](../evaluation/README.md) for other policies.
