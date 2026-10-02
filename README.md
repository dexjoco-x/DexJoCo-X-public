# DexJoCo-X

DexJoCo-X 提供七种代表性灵巧手、六个任务的训练数据与统一仿真评测环境，共 42 个手–任务组合、2,100 条示范，每个组合 50 条。

本候选包含仿真与必要模型资产、数据清单及下载/转换工具、Native / FAAS / DexLatent 评测接口，以及使用自有权重的 π0.5 推理示例。模型训练实现、训练补丁、策略与编码器权重、手套映射和旧 Git 历史均不包含在此版本中。完整视频和张量单独存放在 Hugging Face 数据仓库；代码及数据目前仍保持私有。

## 安装与环境检查

第一次使用请从 [完整快速上手](docs/QUICKSTART.md) 开始，依次安装、下载单任务、查看示范并接入自己的策略。

使用 Python 3.11，在仓库根目录安装：

```bash
pip install -c requirements/constraints-python311.txt -e './simulator[benchmark,dataset,download]'
python tools/runtime_assets.py check
```

无显示器环境需安装 OSMesa，并设置：

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
python tools/portability_smoke.py
python tools/representation_smoke.py --representations native faas
python tools/scope_smoke.py --output ./outputs/diagnostic_replay
```

这些检查覆盖 42 个环境、动作接口、图像和确定性回放。DexLatent 另需安装 `./simulator[dexlatent]` 并提供兼容的本地编码器。

## 下载与使用

完整数据按六任务分包，共 8.242 GB。每个包含七只手各 50 条示范。当前私有下载需要有权访问数据仓库的 Hugging Face 账号。

```bash
hf auth login
export DATA_ROOT=/path/to/DexJoCo-X_dataset
python tools/dataset_packages.py download --tasks hammer_nail --data-root "$DATA_ROOT"
python dataset/tools/prepare_server.py "$DATA_ROOT"
python tools/dataset_smoke.py --data-root "$DATA_ROOT" --output ./outputs/data_check.json
```

下载包包含原生张量、视频、合约、来源及校验信息；转换后的 Parquet 数据另行生成。使用者按这些字段对接自己的训练框架，并用自有 checkpoint 在评测环境中执行。

## 手与任务

| 手型 | 内部键 |
|---|---|
| Allegro | `allegro` |
| Inspire | `inspire` |
| LEAP | `leap` |
| LinkerHand | `linkerhand_l20` |
| Sharpa Wave | `sharpa_wave` |
| Wuji | `wuji` |
| XHand | `xhand` |

LinkerHand 的历史内部键保持兼容，模型源几何来自官方 L25，见 [版本说明](docs/VERSIONING.md)。

单臂任务：`hammer_nail`、`pick_bucket`、`single_hand_hanoi`。
双臂任务：`bimanual_microwave_cook`、`bimanual_photograph`、`bimanual_unlock_ipad`。

每个手–任务组合评估三次，每次 50 个 rollout。评测使用独立场景种子，策略只接收图像、机器人状态、掩码和指令。

## 使用指南

- [数据下载与转换](dataset/DOWNLOAD.md)、[数据字段](dataset/README.md)、[数据卡](docs/DATASET_CARD.md)。
- [读取原始示范](examples/read_demo.py)：输出未归一化的 `arm40_v1` 状态、动作块、掩码与 RGB。
- [仿真环境](simulator/README.md)和[评测协议](evaluation/README.md)。
- [π0.5 推理示例](examples/pi05/README.md)：RLinf 推理后端，支持单臂 32 维和 Ego-π 左右交替排列。
- [场景扩增工具](collection/README.md)：以已有机器人示范为输入，不提供手套录制工具。
- [本候选验收](docs/DATASET_EVAL_RELEASE_VALIDATION.md)和[发布范围](docs/RELEASE_LAYOUT.md)。

## 来源与许可

原创代码采用 [MIT](LICENSE)，适用范围见 [LICENSE_SCOPE.md](LICENSE_SCOPE.md)。第三方代码和模型资产保留各自许可及来源，不统一重新授权为 MIT。完整示范数据的许可另行确认；第三方资产的已记录条款与未确认范围见 [来源说明](THIRD_PARTY_NOTICES.md)、[当前许可状态](LICENSE_STATUS.md)。
