# π₀.₅ integration with DexJoCo-X

[中文](README.md) | **English**

Load a trained checkpoint, statistics, and tokenizer; receive RGB images, states, masks, and instructions; return action chunks in physical units.

This example uses [RLinf's π₀.₅ PyTorch inference implementation](https://github.com/RLinf/RLinf/tree/e1d1bac185fc8d2718c25df9ae8a9089893c46c2/rlinf/models/embodiment/openpi_rlinf), pinned to `e1d1bac185fc8d2718c25df9ae8a9089893c46c2`. Install the upstream code in a separate environment; the entry point loads its inference module directly.

## Action and observation conventions

| Layout | Model action head | Physical steps per chunk | Scope |
|---|---|---|---|
| `single_right32` | H × 32 | H | Three single-right-arm tasks |
| `ego_interleave32` | 2H × 32, left then right | H | Single-arm and bimanual tasks; matches the paper's π₀.₅ (Ego-π) layout |

The example commands use the second layout: 50 action tokens pair into 25 physical time steps. Both layouts use a 32-dimensional action head.

- The evaluator uses `arm40_v1`: right side 0:40 and left side 40:80. Each side begins with six wrist dimensions, followed by hand actuators in their defined order. The model uses right slots 0:32 and left slots 40:72; remaining slots must be inactive.
- States contain world-frame wrist position (meters), rotation vectors (radians), and native hand states. Actions contain world-frame wrist increments (meters/radians) and native actuator targets. See the [environment contracts](../../evaluation/contracts) for actuator names, ordering, and ranges. Execution follows the environment control period.
- Images use RGB uint8 HWC. `base`, `wrist_left`, and `wrist_right` map to model inputs `base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb`. The upstream model resizes 256×256 images to 224×224. Missing camera images and masks are disabled.
- Instructions use `Embodiment: {hand}. {instruction}`. States are normalized with training statistics and discretized into text tokens. An overlong context raises an error.
- Quantile normalization matches the Ego-π interface: ordinary dimensions use q01/q99; spans below 1e-4 use midpoint offsets. Actions are inversely transformed with the same statistics, and invalid dimensions are zeroed.
- The evaluator supplies explicit flow noise. Alternating layouts pair this noise left then right in the same order; the server does not resample it.

## Installation

Use a separate Python 3.11 environment for the policy server. Simulation can run in another environment over WebSocket. Install a PyTorch build supporting your GPU, then install the example dependencies.

```bash
git clone https://github.com/RLinf/RLinf.git /path/to/RLinf
git -C /path/to/RLinf checkout e1d1bac185fc8d2718c25df9ae8a9089893c46c2
pip install -r examples/pi05/requirements.txt
```

The validated GPU configuration uses PyTorch 2.8.0 with CUDA 12.8: `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`. Alternatively, choose a build compatible with your GPU driver.

Provide your own **complete RLinf model state_dict** in `.pt` or `.safetensors` format and the matching PaliGemma tokenizer. OpenPI JAX checkpoints and official PyTorch files are not direct inputs to this entry point. For OpenPI-trained models, use the [upstream RLinf conversion tools](https://github.com/RLinf/RLinf/tree/e1d1bac185fc8d2718c25df9ae8a9089893c46c2/rlinf/utils/ckpt_convertor/openpi) and verify that training and inference match in layout, tokenizer, normalization, and instructions. Weight conversion alone does not turn an ordinary π₀.₅ policy into an alternating-prediction policy.

## Bind the checkpoint interface

Save the quantiles used during training in `statistics.json`: two objects, `state` and `action`, each containing `q01` and `q99` arrays. In alternating mode, both state and action statistics have 80 dimensions. In single-arm mode, state statistics have 32 dimensions and action statistics have 80, with only the right-side model slots active.

Use the checkpoint's training statistics. Evaluation inputs are physical values, unlike the prenormalized Parquet views; see the [dataset guide](../../dataset/README.en.md).

See the [quickstart](../../docs/QUICKSTART.en.md) for reading raw physical fields. The alternating layout pairs left `actions[:, 40:72]` and right `actions[:, :32]` at each time step: 25 steps form 50 tokens.

Run from the repository root. This example describes a six-task model for Sharpa Wave:

```bash
python -m examples.pi05.make_contract \
  --checkpoint /path/to/my_full_weights.pt \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --statistics /path/to/statistics.json \
  --token-layout ego_interleave32 --action-horizon 25 \
  --max-token-len 384 --inference-steps 5 \
  --hands sharpa_wave \
  --tasks hammer_nail pick_bucket single_hand_hanoi \
          bimanual_microwave_cook bimanual_photograph bimanual_unlock_ipad \
  --output ./outputs/my_pi05_contract.json
```

Set `hands` and `tasks` to the checkpoint's actual supported scope. The interface file binds checkpoint/tokenizer SHA-256 hashes, action layout, horizon, statistics, and scope. The server rejects mismatched files and unsupported tasks. For a diagnostic model, add `--diagnostic-only` when generating this file.

## Serve and evaluate

Terminal one: start the policy server.

```bash
python -m examples.pi05.serve \
  --source /path/to/RLinf \
  --checkpoint /path/to/my_full_weights.pt \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --contract ./outputs/my_pi05_contract.json \
  --device cuda --port 8200
```

Terminal two: evaluate in a simulation environment with `./simulator[benchmark]` installed.

```bash
python evaluation/evaluate.py \
  --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand sharpa_wave --task all --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/my_pi05_evaluation
```

Each hand–task pair has three runs of 50 rollouts. Results include success fields, scene and policy seeds, and interface/checkpoint hashes. When serving a model for one hand, select the same hand in the evaluator. Serve and evaluate seven separate hand-specific models individually. For a short diagnostic, use `--episodes 1 --diagnostic --max-steps 16` with a supported task. Diagnostic results are recorded separately from full success-rate evaluations.

For single-arm mode, change the contract command to `--token-layout single_right32` and select only single-arm tasks. Keep the physical `--action-horizon` consistent between server and evaluator.
