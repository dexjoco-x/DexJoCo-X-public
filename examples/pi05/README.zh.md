# 使用 Training v1 训练和评估 π₀.₅

[English](README.md) | **中文**

本示例串起公开数据集、π₀.₅ 微调、策略服务和 benchmark 评测。每次训练一种手，可同时使用全部六个任务。模型使用固定版本的 [RLinf PyTorch 实现](https://github.com/RLinf/RLinf/tree/e1d1bac185fc8d2718c25df9ae8a9089893c46c2/rlinf/models/embodiment/openpi_rlinf)。

## 1. 安装并准备模型

训练和策略服务使用单独的 Python 3.11 环境；仿真使用[参考环境](../../docs/REPLAY_REPAIR.zh-CN.md)，通过 WebSocket 通信。

```bash
git clone https://github.com/RLinf/RLinf.git /path/to/RLinf
git -C /path/to/RLinf checkout e1d1bac185fc8d2718c25df9ae8a9089893c46c2
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r examples/pi05/requirements-training.txt
```

准备完整的 RLinf π₀.₅ 基础权重（`.pt` 或 `.safetensors`）及匹配的 PaliGemma tokenizer。官方基础模型位于 `gs://openpi-assets/checkpoints/pi05_base`，可按 [OpenPI 文档](https://github.com/Physical-Intelligence/openpi#base-models)获取，再使用固定版本的 [RLinf 转换工具](https://github.com/RLinf/RLinf/blob/e1d1bac185fc8d2718c25df9ae8a9089893c46c2/rlinf/utils/ckpt_convertor/openpi/README.md)转换。JAX 和 OpenPI PyTorch 权重不能直接传给此入口。[Tokenizer 下载](https://storage.googleapis.com/big_vision/paligemma_tokenizer.model)。

默认冻结视觉与语言主干，微调动作专家及动作、时间投影层；使用 BF16 计算、FP32 可训练参数、AdamW、cosine 学习率、梯度累积和带掩码的 flow-matching 损失。这是最小训练示例，不是论文实验的训练配置。已在 32 GB 显存的 RTX 5090 上完成短程训练、保存、重新加载和仿真接口验证。

## 2. 下载数据并训练

按[数据集说明](../../dataset/README.zh.md)下载 Training v1。以下命令均在仓库根目录执行：

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

不填写 `--tasks` 时使用六个任务；选择子集可加 `--tasks hammer_nail bimanual_photograph`。每次训练一只手，更换 `--hand` 可分别训练七个模型。从所选示范中均匀采样帧，上述命令的有效 batch size 为 8。

输出包括 `model.safetensors`、`contract.json`、`run.json` 和 `loss.jsonl`。模型在训练结束时保存，接口文件绑定权重、tokenizer、统计量、数据版本、手、任务和动作排列。仅测试流程时，使用 `--steps 2 --accumulation-steps 1 --diagnostic`，并指定新的输出目录。

## 3. 启动策略并评估

在策略环境中运行：

```bash
python -m examples.pi05.serve_training_v1 \
  --source /path/to/RLinf --run ./outputs/pi05_xhand \
  --tokenizer /path/to/paligemma_tokenizer.model --port 8200
```

在参考仿真环境中运行：

```bash
python evaluation/evaluate.py \
  --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand xhand --task all --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/pi05_xhand_evaluation
```

正式评估要求每个手–任务组合 **150 回合，即独立三次、每次 50 回合**。仅检查接口时，选择一个支持的任务，使用 `--episodes 1 --diagnostic --max-steps 16`。标记为调试用途的模型不能生成正式结果。输出会记录代码、数据包、模型、接口、仿真环境和随机种子。

## 数据与动作约定

| 环节 | 处理方式 |
|---|---|
| 数据标签 | 直接读取 Training v1 保存的 float32 标签，不再次归一化 |
| 模型目标 | 50 × 32 个数值：左、右、左、右交替，对应 25 个物理时间步 |
| 损失掩码 | 排除无效关节、不存在的手臂及轨迹末尾补齐的时间步 |
| 状态输入 | 使用冻结的手–任务统计量，离散状态附加到任务文本中 |
| 图像输入 | 前视、左腕、右腕；训练和推理共用 RGB 预处理 |
| 仿真指令 | 使用 Training v1 解码器恢复为 25 × 80 的 float64 物理动作 |

保留 32 维动作头。仿真的 `arm40_v1` 排列中，右手取 0:32，左手取 40:72。评估器提供 flow noise，服务端按相同的左右顺序配对。每次预测 25 步，执行前 8 步后重新预测。读取器直接使用 bundle 中的原始保存标签，与准备好的 LeRobot、Being-H 视图一致；加载 `prepared/beingh_final_v3_train_stats.json`，不重新拟合统计量。

只做单臂任务时，可使用 `--token-layout single_right32`，每块包含 25 个右手 token；该模式拒绝双臂任务。训练与评估的物理动作长度须一致。

使用旧归一化接口训练的模型继续使用 `make_contract.py` 和 `serve.py`，保留原统计量；按本示例新训练的模型使用 `serve_training_v1.py`。
