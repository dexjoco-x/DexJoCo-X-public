# 快速上手

**中文** | [English](QUICKSTART.en.md)

所有命令在仓库根目录执行。先完成 [首页的安装步骤](../README.md#开始使用)。

## 1. 检查环境

```bash
python tools/portability_smoke.py
```

命令检查七手六任务的全部 42 个环境。无显示器的 Ubuntu / Debian 可使用 OSMesa：

```bash
sudo apt-get install libosmesa6
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
python tools/portability_smoke.py
```

## 2. 下载并查看一条示范

```bash
export DATA_ROOT="$PWD/outputs/data"
python tools/dataset_packages.py download --tasks hammer_nail \
  --data-root "$DATA_ROOT"
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
```

一个任务包包含七种手各 50 条示范。读取示例输出状态、动作块和掩码，并将所选帧的 RGB 图像存入一个新的输出目录。

也可以在 Python 中读取：

```python
from examples.read_demo import read_demo

sample = read_demo(data_root, task="hammer_nail", hand="allegro", horizon=25)
state = sample["state"]           # (80,)，物理单位
actions = sample["actions"]      # (25, 80)，物理单位
rgb = sample["images"]["front"]   # uint8 RGB，(256, 256, 3)
```

完整下载使用 `--tasks all`。字段和格式转换见 [数据集文档](../dataset/README.md)。

## 3. 接入策略并评测

按 [π₀.₅ / Ego-π 示例](../examples/pi05/README.md) 启动策略服务，然后在仿真环境运行：

```bash
python evaluation/evaluate.py --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand allegro --task hammer_nail --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/allegro_hammer
```

示例使用支持 Allegro 敲钉子任务的策略，每个动作块预测 25 步、执行前 8 步。评估共三次，每次 50 个 rollout。动作排列和预测长度与策略服务保持一致；其他策略的接入方式见 [评测接口](../evaluation/README.md)。
