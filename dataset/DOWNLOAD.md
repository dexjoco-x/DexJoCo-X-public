# 数据下载、分包与校验

版本：`dataset-v1.0-corrected`。共 7 手、6 任务、2,100 条示范，全部为训练数据。每个任务包包含七只手各 50 条，共 350 条。50 条 XHand 微波炉示范使用修正后的左手安装方向 `xhand_left_mount_rx180_v1`。

数据目前保持私有，下载入口为 [Hugging Face 数据仓库](https://huggingface.co/datasets/darenrenjian/DexJoCo-X)。六个 TAR 任务包已上传到该私有仓库的 `task_packages/v1.0/`，原始文件目录仍保留。下载工具默认获取选定任务的 TAR 包，核对归档与包内文件的 SHA-256 后自动解压。

## 六个任务包

| 任务 | 条数 | TAR 大小（GB，十进制） |
|---|---:|---:|
| hammer_nail | 350 | 0.746 |
| pick_bucket | 350 | 0.962 |
| single_hand_hanoi | 350 | 1.334 |
| bimanual_microwave_cook | 350 | 2.081 |
| bimanual_photograph | 350 | 0.955 |
| bimanual_unlock_ipad | 350 | 2.164 |
| 合计 | 2,100 | 8.242 |

NPZ 张量和 H.264 视频已经压缩，因此使用 TAR 分包，避免重新压缩视频。包内均为普通文件，没有软链接或硬链接条目，不包含手套映射、模型权重、完整仿真资产或准备好的训练缓存。

## 在线按任务下载

在代码仓库根目录安装下载工具，并先使用 `hf auth login` 登录有访问权限的账号：

```bash
pip install -c requirements/constraints-python311.txt \
  -e './simulator[benchmark,dataset,download]'
export DATA_ROOT=/path/to/DexJoCo-X_dataset
python tools/dataset_packages.py download --tasks hammer_nail \
  --data-root "$DATA_ROOT" --report ./outputs/hammer_download.json
```

同时下载多个任务可写 `--tasks hammer_nail pick_bucket`；完整下载使用 `--tasks all`。工具只下载选定任务及必要元数据，完成后逐文件核对 SHA-256。默认归档缓存位于 `DATA_ROOT` 的同级目录 `<DATA_ROOT名称>_archives/task_packages/v1.0/`，也可用 `--archives-root /path/to/cache` 指定缓存根目录。归档会保留以便重用，完整下载并解压约需 16.5 GB，转换缓存另计。可通过 `--workers` 调整下载并行数。

TAR 与原始文件下载均绑定干净数据仓库版本 `9f48e1404168227a22586af0cffb8371dbb7e1f1`，不会随远端 `main` 更新而改变数据。2026-10-02 数据入口已迁移至独立历史，原始示范和任务包的哈希不变；旧历史与权重保留在私有归档中，不作为数据下载入口。查看预计文件数与大小：

```bash
python tools/dataset_packages.py list
python tools/dataset_packages.py download --tasks all \
  --data-root "$DATA_ROOT" --dry-run
```

如希望逐文件下载原始数据而不保存 TAR，可选择：

```bash
python tools/dataset_packages.py download --transport files --tasks hammer_nail \
  --data-root "$DATA_ROOT"
```

`--transport archive` 明确选择归档方式，默认 `auto` 使用本索引中已上传的归档。当前私有数据访问需要授权；令牌仅由登录缓存或 `HF_TOKEN` 环境变量提供，不写入索引、脚本或 Git。

## 从本地 TAR 包安装

拿到 TAR 文件和随附的 `PACKAGES.json`、`SHA256SUMS` 后，可以先在包目录运行 `sha256sum -c SHA256SUMS`。推荐用代码工具解压，工具会检查整个包的哈希、允许的路径、文件类型和每个文件的哈希：

```bash
export PACK_ROOT=/path/to/task_packages
python tools/dataset_packages.py extract --tasks all \
  --index "$PACK_ROOT/PACKAGES.json" --archives-root "$PACK_ROOT" \
  --data-root "$DATA_ROOT" --report ./outputs/extract_report.json
```

只解压一个任务可改为 `--tasks hammer_nail`。已经存在且哈希一致的数据会保留；有冲突时停止，不覆盖。错误包、链接条目或越界路径会被拒绝。

## 安装后的目录和校验

```text
DATA_ROOT/
  bundle/
    catalog.json
    manifests/<已安装任务>.json
    episodes/<任务>/<手型>/<场景>/
      native.npz
      contract.json
      front.mp4
      wrist_right.mp4
      wrist_left.mp4       # 双臂任务
    evidence/              # 相机检查与 XHand 修正证据
  metadata/                # 完整候选数据的历史审计和相机说明
```

`catalog.json` 和历史审计描述完整基准；仅下载部分任务时，实际可用范围以本地 manifest 和本次校验报告为准。`source` 字段是来源标识，不是使用这些张量与视频时必须存在的本地目录。

```bash
python tools/dataset_packages.py verify --tasks all \
  --data-root "$DATA_ROOT" --report ./outputs/data_verify.json
```

检查单任务时选择相应任务。完整六任务安装覆盖 11,513 个不同文件，其中张量、控制定义和视频共有 9,450 个；其余为相机检查、修正证据、索引和元数据。

## 生成训练格式

原始数据是唯一的张量和视频来源，三种训练视图另行生成，不重复放入下载包。使用已下载任务对应的 manifest 生成 Native 和 FAAS：

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT"
python dataset/tools/prepare_benchmark7_repr.py "$DATA_ROOT" \
  --representation faas \
  --contracts representations/table1_sided_v3/codecs/faas7_project_v3_sided.json
```

DexLatent 另需安装匹配的 PyTorch，并提供自己的兼容编码器，或使用有权使用的私有匹配编码器及其冻结记录：

```bash
python dataset/tools/prepare_benchmark7_repr.py "$DATA_ROOT" \
  --representation dexlatent \
  --contracts representations/table1_sided_v3/codecs/faas7_project_v3_sided.json \
  --checkpoint /path/to/private/selected.pt \
  --seal representations/table1_sided_v3/codecs/frozen_seal.json
```

上述冻结记录只适用于其绑定的历史编码器；自行训练的新编码器必须使用对应的控制定义与选择记录，不应冒用历史哈希。源仓库不提供编码器或策略权重。

输出目录为 `prepared/beingh05/`、`prepared/beingh05_faas/`、`prepared/beingh05_dexlatent/`。完整下载后分别覆盖 42 组、2,100 条、946,156 帧；部分任务转换只覆盖已安装的任务。已有 FAAS/DexLatent 输出时工具停止，请使用新的工作目录。跨文件系统的视频会复制，同一文件系统内优先使用硬链接。

## 数据接口与推理验收

转换 Native 后，检查已下载的所有示范与转换结果：

```bash
python tools/dataset_smoke.py --data-root "$DATA_ROOT" --output ./outputs/data_check.json
```

使用自有 π0.5 checkpoint 的安装、接口文件与运行命令见 [推理示例](../examples/pi05/README.md)。模型训练由使用者选用的上游框架完成；本仓库只提供数据转换、推理适配和评测代码。

本候选从空目录安装、下载与仿真调用的结果见 [验收记录](../docs/DATASET_EVAL_RELEASE_VALIDATION.md)。
