# Train and evaluate π₀.₅ on Training v1

**English** | [中文](README.zh.md)

This example connects the released dataset to π₀.₅ fine-tuning, model serving, and benchmark evaluation. It trains one hand on any of the six tasks using the pinned [RLinf PyTorch backend](https://github.com/RLinf/RLinf/tree/e1d1bac185fc8d2718c25df9ae8a9089893c46c2/rlinf/models/embodiment/openpi_rlinf).

## 1. Install and prepare the model

Use a separate Python 3.11 environment for training and serving. The simulator runs in the [reference environment](../../docs/REPLAY_REPAIR.md) and communicates over WebSocket.

```bash
git clone https://github.com/RLinf/RLinf.git /path/to/RLinf
git -C /path/to/RLinf checkout e1d1bac185fc8d2718c25df9ae8a9089893c46c2
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r examples/pi05/requirements-training.txt
```

Provide a complete RLinf π₀.₅ base state dictionary (`.pt` or `.safetensors`) and its PaliGemma tokenizer. The official base is `gs://openpi-assets/checkpoints/pi05_base`; obtain it through [OpenPI](https://github.com/Physical-Intelligence/openpi#base-models) and convert it with the pinned [RLinf checkpoint converter](https://github.com/RLinf/RLinf/blob/e1d1bac185fc8d2718c25df9ae8a9089893c46c2/rlinf/utils/ckpt_convertor/openpi/README.md). JAX and OpenPI PyTorch files require conversion before loading here. The tokenizer is available at [Google's PaliGemma tokenizer URL](https://storage.googleapis.com/big_vision/paligemma_tokenizer.model).

The default recipe freezes the vision/language backbone and fine-tunes the action expert and action/time projections. It uses BF16 computation, FP32 trainable parameters, AdamW, cosine decay, gradient accumulation, and masked flow-matching loss. This is a minimal starting recipe, not the paper's training configuration. A short train–save–reload–evaluate check ran on an RTX 5090 with 32 GB VRAM.

## 2. Download the dataset and train

Download [Training v1](../../dataset/README.md) using the dataset tools. Run these commands from the repository root:

```bash
export DATA_ROOT="$PWD/outputs/training_v1"
python tools/dataset_packages.py download --tasks all --data-root "$DATA_ROOT"

python -m examples.pi05.train \
  --source /path/to/RLinf \
  --checkpoint /path/to/pi05_base_rlinf/model.safetensors \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --dataset-root "$DATA_ROOT" --hand xhand \
  --steps 10000 --batch-size 1 --accumulation-steps 8 \
  --output ./outputs/pi05_xhand
```

Omitting `--tasks` selects all six tasks. To train a subset, add e.g. `--tasks hammer_nail bimanual_photograph`. Each run trains one hand; repeat with other hands for seven hand-specific models. Frames are sampled uniformly from the selected demonstrations. The effective batch size above is 8.

The output contains `model.safetensors`, `contract.json`, `run.json`, and `loss.jsonl`. The model is saved at the end of the run. The contract binds the checkpoint, tokenizer, frozen statistics, dataset version, hand, tasks, and action layout. Short connection tests should use `--steps 2 --accumulation-steps 1 --diagnostic` and a separate output directory.

## 3. Serve and evaluate

In the policy environment:

```bash
python -m examples.pi05.serve_training_v1 \
  --source /path/to/RLinf --run ./outputs/pi05_xhand \
  --tokenizer /path/to/paligemma_tokenizer.model --port 8200
```

In the reference simulation environment:

```bash
python evaluation/evaluate.py \
  --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand xhand --task all --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/pi05_xhand_evaluation
```

Formal evaluation requires **150 episodes per hand–task pair: three independent repeats of 50**. For a connection check, select one supported task and use `--episodes 1 --diagnostic --max-steps 16`. A diagnostic checkpoint cannot produce formal results. Results record code, benchmark archive, checkpoint, interface, environment, and seed information.

## Data and action conventions

| Stage | Representation |
|---|---|
| Saved targets | The exact float32 labels in Training v1; no second normalization |
| Model targets | 50 × 32 tokens: left, right, left, right … for 25 physical steps |
| Loss mask | Excludes inactive joints, missing arms, and steps beyond the episode end |
| Model state | Frozen per-hand/task normalization; state bins appended to the instruction |
| Model images | Front, left wrist, right wrist; RGB preprocessing shared by training and serving |
| Simulator commands | The frozen Training v1 decoder restores 25 × 80 physical commands in float64 |

The 32-dimensional action head is retained. Right-hand channels use slots 0:32 and left-hand channels 40:72 in the simulator's `arm40_v1` layout. The evaluator supplies the flow noise; the server pairs it in the same left/right order. Each inference predicts 25 physical steps and executes eight before re-inference. The reader uses the bundle's saved labels, which match the prepared LeRobot and Being-H views, and loads `prepared/beingh_final_v3_train_stats.json` without fitting new statistics.

For single-arm-only experiments, `--token-layout single_right32` uses 25 right-hand tokens. It rejects bimanual tasks. Keep the physical horizon consistent in training and evaluation.

Earlier checkpoints using the old quantile interface retain `make_contract.py` and `serve.py`. Their statistics must not be replaced with Training v1 statistics; newly trained models above use `serve_training_v1.py`.
