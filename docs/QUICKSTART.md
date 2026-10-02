# 从下载示范到仿真评测

本仓库已包含七手六任务所需的全部仿真模型、网格和纹理，无需到其他项目另行获取模型。以下下载步骤仅用于本项目的示范数据。

所有命令在代码仓库根目录运行。当前代码与数据仓库均为私有，需要有访问权限的 GitHub 和 Hugging Face 账号。

## 1. 安装仿真与数据环境

```bash
git clone git@github.com:darenrenjian/DexJoCo-X-public.git
cd DexJoCo-X-public
python3.11 -m venv .venv
source .venv/bin/activate
pip install -c requirements/constraints-python311.txt \
  -e './simulator[benchmark,dataset,download]'
python tools/runtime_assets.py check
```

无显示器的 Ubuntu/Debian 环境可安装 `libosmesa6`，随后设置渲染方式：

```bash
sudo apt-get install libosmesa6
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
python tools/portability_smoke.py
```

此检查覆盖七手六任务的 42 个环境组合，不需要策略 checkpoint。

## 2. 下载一个任务并读取示范

```bash
hf auth login
export DATA_ROOT="$PWD/outputs/data"
python tools/dataset_packages.py download --tasks hammer_nail \
  --data-root "$DATA_ROOT" --report ./outputs/download.json
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
```

一个任务包包含七手各 50 条示范。示例打印状态、动作块、有效维度与指令，并导出所选帧的 RGB 图片。输出目录须为新目录。它直接读取原始张量，保留物理单位；无需先生成 Parquet。

在自己的数据接口中读取：

```python
from examples.read_demo import read_demo

sample = read_demo(data_root, task="hammer_nail", hand="allegro", horizon=25)
state = sample["state"]           # (80,)，未归一化
actions = sample["actions"]       # (25, 80)，未归一化
action_mask = sample["action_mask"]
rgb = sample["images"]["front"]   # uint8 RGB，(256, 256, 3)
```

| 字段 | 顺序与含义 |
|---|---|
| 每侧 40 维 | 右侧为 0:40，左侧为 40:80；腕部占前 6 维，之后为原生手通道 |
| 状态腕部 | 世界位置（米）与旋转向量（弧度）；原始 NPZ 的四元数在读取时转换 |
| 动作腕部 | 世界坐标位移/旋转增量（米/弧度）；手通道为 actuator 目标值 |
| 掩码 | 标识活动侧和有效手通道；轨迹尾部补齐的动作步骤全部无效 |

手通道顺序与范围见对应 episode 的 `contract.json`。转换后的腕部状态采用 rotvec，避免将原始四元数的 7 个腕部字段直接输入评测接口。

π0.5（Ego-π）的左右交替动作块可以从物理动作中排列得到：

```python
import numpy as np

tokens = np.stack((actions[:, 40:72], actions[:, :32]), axis=1).reshape(-1, 32)
token_mask = np.stack((action_mask[:, 40:72], action_mask[:, :32]), axis=1).reshape(-1, 32)
# 25 个物理时间步 → 50 个 32 维 token；每一对先左后右。
```

缺失侧通过掩码关闭，仍保留其 token 位置。训练时应使用与 checkpoint 一致的归一化、图像处理和指令格式；这里仅演示物理字段读取和动作排列。

## 3. 按需生成通用训练视图

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT"
python tools/dataset_smoke.py --data-root "$DATA_ROOT" \
  --output ./outputs/data_check.json
```

Native 的 Parquet 输出位于 `prepared/beingh05/`，名称为历史兼容键，含 200 维预归一化数值。这与上一步的 80 维物理数据不同，接入训练框架时需要明确选择，并保持训练与推理的归一化一致。FAAS 转换及完整六任务下载见 [数据指南](../dataset/DOWNLOAD.md)。本仓库不包含模型训练实现。

## 4. 用自己的 π0.5 checkpoint 评测

策略服务使用独立 Python 环境，并在外部安装固定版本的 RLinf。按 [π0.5 接入指南](../examples/pi05/README.md) 完成依赖安装、checkpoint 格式核对和接口文件生成，然后启动服务。

在本页的仿真环境中先做短验收：

```bash
python evaluation/evaluate.py --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand allegro --task hammer_nail --episodes 1 \
  --action-horizon 25 --execution-horizon 8 \
  --diagnostic --max-steps 16 --output ./outputs/pi05_diagnostic
```

该命令要求服务支持 Allegro 敲钉子任务。正式评测移除 `--diagnostic --max-steps 16`，改为 `--episodes 150`；六任务模型可使用 `--task all`。每个手–任务组合执行三次独立 50-rollout 评测，详见 [评测协议](../evaluation/README.md)。仅诊断用途的 checkpoint 无法生成正式结果。

本项目不提供 checkpoint、tokenizer 或编码器权重。已完成的环境、下载、转换与实际策略接入验收见 [验收记录](DATASET_EVAL_RELEASE_VALIDATION.md)。
