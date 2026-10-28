# 数据集

[English](README.md) | **中文**

**Training v1 是当前统一的 benchmark 数据集，训练和示范回放都使用这一版。**

7 种手 · 6 个任务 · 2,100 条训练示范 · 933,865 帧 · 20 Hz。
每个手–任务组合有 50 条示范。float64 动作、LeRobot 与 Being-H 格式对应同一批轨迹，使用相同的视频。

## 下载

完成[安装](../README.zh.md#开始使用)后运行：

```bash
export DATA_ROOT="$PWD/outputs/training_v1"
python tools/dataset_packages.py download --tasks all --data-root "$DATA_ROOT"
```

工具下载一个 5.21 GB 的包，检查整包 SHA-256，解压后核对所有文件。
建议预留 19 GB 空间。包内包含全部六个任务和两种训练格式；旧命令即便通过
`--tasks` 指定单个任务，也会下载完整包。加上 `--dry-run` 可先查看下载内容。

归档默认缓存在数据目录旁的 `training_v1_archives/training_v1/`，可通过
`--archives-root` 修改。版本和哈希固定在 [`PACKAGES.json`](PACKAGES.json) 中。
已有的其他版本数据不会被覆盖；请使用新的目录。

已经下载 Training v1 归档时：

```bash
python tools/dataset_packages.py extract --archives-root /path/to/training_v1 \
  --data-root "$DATA_ROOT"
```

## 选择读取方式

| 用途 | 解压后的路径 |
|---|---|
| 回放示范、读取物理单位数据 | `bundle/` |
| 官方 LeRobot v3，已用 0.4.4 验证 | `prepared/lerobot_final_v3/` |
| Being-H 读取器 | `prepared/beingh_final_v3/` |
| 对应的归一化与解码统计量 | `prepared/*_train_stats.json` |

它们是同一版数据的不同读取格式，不是两套数据。训练格式已经准备好，无需自行转换。
float32 标签已经归一化，推理使用附带的统计量和 `Native200PolicyAdapter`，
发送到仿真环境时保留解码后的 float64 精度。详见[训练使用说明](../docs/TRAINING_CONVERSION.zh.md)。

## 文件与单位

```text
bundle/
  catalog.json
  manifests/<任务>.json
  episodes/<任务>/<手型>/<场景>/
    native.npz
    contract.json
    front.mp4
    wrist_right.mp4
    wrist_left.mp4       # 双臂任务
prepared/
  lerobot_final_v3/train/<任务>/<手型>/
  beingh_final_v3/train/<任务>/<手型>/
metadata/VERIFICATION.json
SHA256SUMS
```

manifest 记录场景种子、有效步数和文件哈希。`native.npz` 保存状态、float64
物理动作、float32 训练标签、掩码和时间戳。RGB 为 256×256、20 Hz，每帧观测对应执行动作之前的状态。

原始数组先右侧、后左侧。每侧状态以世界位置和 wxyz 四元数开头；动作前六维是世界坐标下
腕部位移与旋转增量，后接手部执行器目标值。位置单位为米，旋转单位为弧度。
手通道顺序及范围见 `contract.json`。原始相机掩码顺序为 `[front, wrist_right, wrist_left]`，
策略接口为 `[base, wrist_left, wrist_right]`，其中 `base` 对应 `front`，请按名称匹配。

```bash
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
python tools/dataset_packages.py verify --data-root "$DATA_ROOT"
```

示例读取器返回物理单位的 80 维状态/动作、掩码和 RGB。参见[回放检查](../docs/REPLAY_REPAIR.zh-CN.md)、
[策略评估](../evaluation/README.zh.md)和[许可范围](LICENSE_SCOPE.md)。
旧版只保留在[历史记录](../docs/VERSIONING.zh.md)中，新实验统一使用 Training v1。
