# DexJoCo-X

用于比较不同灵巧手动作表征的多手操作基准。

[项目网站](https://darenrenjian.github.io/DexJoCo-X-website/) · [论文](https://arxiv.org/abs/2610.03278) · [数据集](https://huggingface.co/datasets/darenrenjian/DexJoCo-X) · [English](README.en.md)

![DexJoCo-X 项目总览](https://darenrenjian.github.io/DexJoCo-X-website/assets/overview.webp)

**7 种灵巧手 · 6 个任务 · 42 个手–任务组合 · 2,100 条示范**

DexJoCo-X 将多种灵巧手置于统一的任务、数据格式和评测接口中，支持比较 Native、FAAS 和 DexLatent 动作表征。每个手–任务组合包含 50 条训练示范，闭环评测使用独立场景种子。

## 项目内容

- **数据集**：20 Hz 示范，包含 RGB 视频、机器人状态、动作、有效维掩码和任务合约。
- **仿真环境**：七手六任务所需的机器人模型、物体、网格和纹理随仓库提供，无需另行下载仿真资产。
- **评测工具**：统一策略接口、成功判定和结果记录，提供 π₀.₅ / Ego-π 推理接入示例及已有示范的场景扩增工具。

训练可使用自己的框架；评测需自备策略权重，DexLatent 还需匹配的本地编码器。手套映射与录制工具不包含在本仓库中。

**灵巧手**：Allegro、Inspire、LEAP、LinkerHand、Sharpa Wave、Wuji、XHand。

| 类型 | 任务 | 配置名称 |
|---|---|---|
| 单臂 | 敲钉子 | `hammer_nail` |
| 单臂 | 提桶 | `pick_bucket` |
| 单臂 | 汉诺塔 | `single_hand_hanoi` |
| 双臂 | 微波炉操作 | `bimanual_microwave_cook` |
| 双臂 | 拍照 | `bimanual_photograph` |
| 双臂 | 解锁平板 | `bimanual_unlock_ipad` |

## 安装

使用 Python 3.11：

```bash
git clone https://github.com/darenrenjian/DexJoCo-X-public.git
cd DexJoCo-X-public
python3.11 -m venv .venv
source .venv/bin/activate
pip install -c requirements/constraints-python311.txt \
  -e './simulator[benchmark,dataset,download]'
python tools/runtime_assets.py check
```

无显示器的 Ubuntu / Debian 环境可使用 OSMesa：

```bash
sudo apt-get install libosmesa6
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
python tools/portability_smoke.py
```

环境检查覆盖全部 42 个手–任务组合。完整步骤见 [快速上手](docs/QUICKSTART.md)。

## 下载并查看示范

数据无需登录即可下载。六个任务包共约 **8.24 GB**，每包包含七种手各 50 条示范；完整下载并解压约需 16.5 GB 空间。

以下命令下载敲钉子任务，校验文件并查看一条 Allegro 示范：

```bash
export DATA_ROOT="$PWD/outputs/data"
python tools/dataset_packages.py download --tasks hammer_nail \
  --data-root "$DATA_ROOT"
python -m examples.read_demo --data-root "$DATA_ROOT" \
  --task hammer_nail --hand allegro --frame 0 --horizon 25 \
  --output ./outputs/allegro_preview
```

使用 `--tasks all` 下载全部任务。读取示例输出物理单位下的 80 维状态、动作块、掩码与 RGB；也可按需转换为 Parquet 训练视图。字段、归一化和转换方式见 [数据接口](dataset/README.md)与[下载指南](dataset/DOWNLOAD.md)。

## 接入策略与评测

策略接收图像、机器人状态、掩码和任务指令，返回动作块。按 [π₀.₅ / Ego-π 示例](examples/pi05/README.md)启动自己的策略服务后，可评估其支持的手型和任务，例如：

```bash
python evaluation/evaluate.py --server ws://127.0.0.1:8200 \
  --layout arm40_v1 --representation native \
  --hand allegro --task hammer_nail --episodes 150 \
  --action-horizon 25 --execution-horizon 8 \
  --output ./outputs/allegro_hammer
```

每个手–任务组合执行 **3 次独立评估，每次 50 个 rollout**。其他策略可通过相同接口接入；动作排列、表征和预测长度应与策略服务一致，详见 [评测协议](evaluation/README.md)。

## 文档

| 内容 | 入口 |
|---|---|
| 从安装到策略接入 | [快速上手](docs/QUICKSTART.md) |
| 数据字段与格式转换 | [数据接口](dataset/README.md) · [下载指南](dataset/DOWNLOAD.md) |
| 仿真环境与评测协议 | [仿真环境](simulator/README.md) · [评测接口](evaluation/README.md) |
| π₀.₅ / Ego-π 推理 | [接入示例](examples/pi05/README.md) |
| 自动场景扩增 | [采集工具](collection/README.md) |
| 数据与模型版本 | [版本说明](docs/VERSIONING.md) |

## 引用与许可

引用信息见 [BibTeX](CITATION.bib) 或 [CITATION.cff](CITATION.cff)。

原创代码采用 [MIT](LICENSE)，适用范围见 [代码许可说明](LICENSE_SCOPE.md)。原创数据贡献采用 **CC BY-NC-SA 4.0**，详见 [数据许可](dataset/LICENSE_SCOPE.md)。第三方代码和模型资产保留各自条款；来源、许可及未确认范围见 [第三方说明](THIRD_PARTY_NOTICES.md)与[许可状态](LICENSE_STATUS.md)。
