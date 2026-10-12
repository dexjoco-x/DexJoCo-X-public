# π0.5 接入 DexJoCo-X 评测

**中文** | [English](README.en.md)

读取训练好的 checkpoint、统计量和 tokenizer，接收 RGB、状态、掩码及指令，返回物理单位下的动作块。

示例使用 [RLinf 的 π0.5 PyTorch 推理实现](https://github.com/RLinf/RLinf/tree/e1d1bac185fc8d2718c25df9ae8a9089893c46c2/rlinf/models/embodiment/openpi_rlinf)，固定版本为 `e1d1bac185fc8d2718c25df9ae8a9089893c46c2`。上游模型代码在独立环境中安装，入口直接加载其推理模块。

## 动作与观测约定

| 排列 | 网络动作头 | 每块物理步数 | 使用范围 |
|---|---|---|---|
| `single_right32` | H × 32 | H | 三个单右臂任务 |
| `ego_interleave32` | 2H × 32，先左后右 | H | 单臂、双臂任务；与论文的 π0.5（Ego-π）排列一致 |

默认示范命令采用第二种排列：50 个动作 token 配对成 25 个物理时间步。两种排列均使用 32 维动作头。

- 评测器采用 `arm40_v1`：右侧 0:40，左侧 40:80。每侧前 6 维为腕部，后面为手的 actuator 顺序；模型使用右侧 0:32、左侧 40:72，其余槽位必须为空。
- 状态为世界坐标腕部位置（米）、旋转向量（弧度）及原生手状态。动作是世界坐标腕部增量（米/弧度）与原生 actuator 目标值；具体 actuator 名称、顺序及范围见对应的 [环境合约](../../evaluation/contracts)。运行时按环境控制周期执行。
- 图像为 RGB uint8 HWC，`base`、`wrist_left`、`wrist_right` 分别映射到模型的 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。上游模型将 256×256 图像缩放为 224×224；缺失相机图像和掩码均关闭。
- 指令格式为 `Embodiment: {hand}. {instruction}`。状态按训练统计量归一化后离散为文本 token；上下文超长会报错。
- 示例的分位数归一化与 Ego-π 接口一致：一般维度按 q01/q99 映射，跨度小于 1e-4 时使用中点偏移。动作按相同统计量反变换，并将无效维度归零。
- 评测器提供显式流噪声。交替排列对噪声使用相同的先左后右配对，服务不会重新抽样。

## 安装

策略服务使用单独的 Python 3.11 环境；仿真可以使用另一个环境，两者通过 WebSocket 连接。先按显卡安装支持该 GPU 的 PyTorch，再安装示例依赖。

```bash
git clone https://github.com/RLinf/RLinf.git /path/to/RLinf
git -C /path/to/RLinf checkout e1d1bac185fc8d2718c25df9ae8a9089893c46c2
pip install -r examples/pi05/requirements.txt
```

已验证的 GPU 配置为 PyTorch 2.8.0 + CUDA 12.8，可运行 `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`，或选择与显卡驱动兼容的 PyTorch 构建。

准备自己的 **完整 RLinf 模型 state_dict**，格式为 `.pt` 或 `.safetensors`，以及对应的 PaliGemma tokenizer。OpenPI 的 JAX checkpoint 或官方 PyTorch 文件不是此入口的直接输入；使用 OpenPI 训练时，需按 [RLinf 上游转换工具](https://github.com/RLinf/RLinf/tree/e1d1bac185fc8d2718c25df9ae8a9089893c46c2/rlinf/utils/ckpt_convertor/openpi) 转换，并验证训练和推理的排列、tokenizer、归一化与指令一致。仅转换权重不会自动把普通 π0.5 策略变成交替预测策略。

## 绑定 checkpoint 的接口

把训练时实际使用的分位数保存为 `statistics.json`，结构为 `state`、`action` 两个对象，各含 `q01`、`q99` 数组。交替模式的状态和动作统计量均为 80 维；单臂模式的状态统计量为 32 维，动作统计量为 80 维，其中只有右侧模型槽位有效。

使用 checkpoint 训练时的统计量。评测接口输入为物理值，与预归一化的 Parquet 视图区分；数据字段见 [数据集文档](../../dataset/README.md)。

读取原始物理字段见 [快速上手](../../docs/QUICKSTART.md)。交替排列将左侧 `actions[:, 40:72]` 与右侧 `actions[:, :32]` 按时间配对，25 步形成 50 个 token。

在仓库根目录运行，例如一个使用 Sharpa Wave 的六任务模型：

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

`hands` 和 `tasks` 必须填写 checkpoint 实际支持的范围。接口文件绑定 checkpoint、tokenizer 的 SHA-256、动作排列、预测长度、统计量与支持范围。服务拒绝不匹配的文件和范围外任务。用于流程验收的模型在生成接口文件时加 `--diagnostic-only`。

## 启动与评估

终端一，启动策略：

```bash
python -m examples.pi05.serve \
  --source /path/to/RLinf \
  --checkpoint /path/to/my_full_weights.pt \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --contract ./outputs/my_pi05_contract.json \
  --device cuda --port 8200
```

终端二，在安装了 `./simulator[benchmark]` 的仿真环境中评估：

```bash
python evaluation/evaluate.py \
  --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand sharpa_wave --task all --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/my_pi05_evaluation
```

每个手–任务组合评估三次、每次 50 条；结果保存成功字段、场景和策略种子、接口及 checkpoint 哈希。服务为一个指定手型的模型时，评测器也选择该手型；七手独立模型分别启动和评估。短验收使用 `--episodes 1 --diagnostic --max-steps 16`，同时选择 checkpoint 支持的一个任务。诊断结果与正式成功率分开记录。

单臂模式在上述接口生成命令中改为 `--token-layout single_right32`，只选择单臂任务；服务和评测器使用一致的物理 `--action-horizon`。
