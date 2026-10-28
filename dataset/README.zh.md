# 数据集：下载、读取与转换

[English](README.md) | **中文**

**7 种手 · 6 个任务 · 2,100 条训练示范 · 939,356 帧 · 20 Hz**

[Hugging Face 数据集](https://huggingface.co/datasets/darenrenjian/DexJoCo-X)按任务提供六个 TAR 包，每包包含七种手各 50 条示范。版本为 `dataset-v1.1-native-replay`。

## 下载

完成 [安装](../README.zh.md#开始使用) 后，在仓库根目录运行：

```bash
export DATA_ROOT="$PWD/outputs/data"
python tools/dataset_packages.py download --tasks hammer_nail \
  --data-root "$DATA_ROOT"
```

使用 `--tasks all` 下载全部任务，或填写多个任务名称。工具自动解压并核对 SHA-256。六个包合计 4.379 GB，下载并解压约占 8.8 GB。

| 任务 | 条数 | 包大小（GB） |
|---|---:|---:|
| `hammer_nail` | 350 | 0.386 |
| `pick_bucket` | 350 | 0.522 |
| `single_hand_hanoi` | 350 | 0.714 |
| `bimanual_microwave_cook` | 350 | 1.108 |
| `bimanual_photograph` | 350 | 0.506 |
| `bimanual_unlock_ipad` | 350 | 1.143 |

常用选项：`--dry-run` 预览下载范围，`--workers` 设置并行数，`--archives-root` 设置归档缓存，`--transport files` 按原始文件下载。默认归档保存在数据目录旁的 `<目录名>_archives/task_packages/v1.1/`。

已有本地任务包时：

```bash
export PACK_ROOT=/path/to/task_packages
python tools/dataset_packages.py extract --tasks all \
  --index "$PACK_ROOT/PACKAGES.json" --archives-root "$PACK_ROOT" \
  --data-root "$DATA_ROOT"
```

下载工具使用 [`PACKAGES.json`](PACKAGES.json) 中固定的提交；数据卡更新不会改变下载内容。

## 目录与原始字段

```text
DATA_ROOT/bundle/
  manifests/<任务>.json
  episodes/<任务>/<手型>/<场景>/
    native.npz
    contract.json
    front.mp4
    wrist_right.mp4
    wrist_left.mp4       # 双臂任务
```

manifest 记录场景、种子、有效步数与文件哈希。`native.npz` 包含 `state`、`action`、状态/动作掩码、`arm_mask`、`timestamp` 和 `image_mask`。

原始 `image_mask` 的顺序为 `[front, wrist_right, wrist_left]`；策略接口的顺序为 `[base, wrist_left, wrist_right]`，其中 `base` 对应 `front`。请按相机名称对应图像。

v1.1 的动作为 float64，精确回放时应保留这些数值。训练转换使用 float32 归一化和裁剪，转换后的动作已不是原始保存指令；详见[精度诊断](../docs/REPLAY_REPAIR.zh-CN.md#数据验证与动作精度)。

原始数组侧序为右、左。每侧状态前七维是腕部世界位置与 wxyz 四元数；动作前六维是世界坐标腕部位移和旋转增量，后面是手部执行器目标值。位置单位为米，旋转单位为弧度；手通道顺序和范围见该示范的 `contract.json`。视频为 256×256、20 Hz 的 RGB，使用 manifest 中的有效步数。

## 读取示范

```bash
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
```

`examples.read_demo.read_demo` 可在 Python 中直接读取。接口将腕部状态转换为旋转向量，返回物理单位下的 80 维状态、动作块、掩码和 RGB；右侧占 0:40，左侧占 40:80。示例见 [快速上手](../docs/QUICKSTART.zh.md)。

## 转换训练格式

需要 Parquet 与视频训练视图时运行：

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT"
python dataset/tools/prepare_benchmark7_repr.py "$DATA_ROOT" --representation faas \
  --contracts representations/table1_sided_v3/codecs/faas7_project_v3_sided.json
```

默认控制坐标和 FAAS 分别输出到 `prepared/beingh05/`、`prepared/beingh05_faas/`。目录名为兼容键；两种视图均为通用数据格式，state/action 宽度为 200，右侧占 0:40，左侧占 40:80，其余填零。腕部状态采用 rotvec，FAAS 手部采用 32 个功能槽位。

转换结果按每个手–任务组合的 q01/q99 归一化并裁剪到 [-1, 1]，无效维度置零。它与读取接口的 80 维物理数据不同；训练和推理采用同一套归一化规则。π₀.₅ 的统计量约定见 [策略示例](../examples/pi05/README.zh.md)。

DexLatent 转换需要兼容的编码器及其匹配记录：

```bash
python dataset/tools/prepare_benchmark7_repr.py "$DATA_ROOT" --representation dexlatent \
  --contracts representations/table1_sided_v3/codecs/faas7_project_v3_sided.json \
  --checkpoint /path/to/selected.pt \
  --seal representations/table1_sided_v3/codecs/frozen_seal.json
```

`--seal` 应绑定所用编码器及控制定义。输出为 `prepared/beingh05_dexlatent/`，已有转换目录时使用新的工作目录。

## 校验与相关说明

```bash
python tools/dataset_packages.py verify --tasks hammer_nail --data-root "$DATA_ROOT"
python tools/dataset_smoke.py --data-root "$DATA_ROOT" --output ./outputs/data_check.json
```

部分下载时选择已安装的任务。完整数据的版本关系、图像配置和许可分别见 [版本说明](../docs/VERSIONING.zh.md)、[相机配置](CAMERA_PROFILE.md)和[数据许可](LICENSE_SCOPE.md)。
