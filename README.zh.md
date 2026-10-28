# DexJoCo-X

[English](README.md) | **中文**

面向多种灵巧手的操作数据集与统一仿真评测环境。

[项目网站](https://jxw-wei.github.io/DexJoCo-X-website/) · [论文](https://arxiv.org/abs/2610.03278) · [数据集](https://huggingface.co/datasets/darenrenjian/DexJoCo-X)

## 更新动态

- 🔧 **2026-10-28** — 发布训练版v1：float32标签、同步视频与状态，以及LeRobot/Being-H预转换格式；2100条最终标签全部通过回放。[下载与用法](docs/TRAINING_CONVERSION.zh.md)。

- 🔧 **2026-10-28** — Runtime v10.1 补充示范字段验证、策略评测数值环境记录与动作精度诊断。详见[回放说明](docs/REPLAY_REPAIR.zh-CN.md)。

- 🔧 **2026-10-27** — 数据集 v1.1 与运行环境 v10 提供通过回放验证的示范和固定数值环境，见[更新说明](docs/REPLAY_REPAIR.zh-CN.md)。

- 🎉 **2026-10-12** — 评测代码与仿真环境现已在 [GitHub](https://github.com/dexjoco-x/DexJoCo-X-public) 公开。
- 🎉 **2026-10-05** — 包含 2,100 条示范的数据集在 [Hugging Face](https://huggingface.co/datasets/darenrenjian/DexJoCo-X) 公开。
- 🎉 **2026-10-02** — 论文首个版本提交至 [arXiv](https://arxiv.org/abs/2610.03278)。

![项目总览](https://jxw-wei.github.io/DexJoCo-X-website/assets/overview.webp)

**7 种灵巧手 · 6 个任务 · 2,100 条示范**

DexJoCo-X 提供操作示范、MuJoCo 仿真环境和统一评测接口。使用数据集训练自己的策略，再接入环境，在一致的任务设置和成功标准下评估多手操作能力。机器人和物体模型、网格、纹理随代码提供。

灵巧手包括 Allegro、Inspire、LEAP、LinkerHand、Sharpa Wave、Wuji、XHand。每个手–任务组合有 50 条训练示范。

| 类型 | 任务 | 配置名称 |
|---|---|---|
| 单臂 | 敲钉子 | `hammer_nail` |
| 单臂 | 提桶 | `pick_bucket` |
| 单臂 | 汉诺塔 | `single_hand_hanoi` |
| 双臂 | 微波炉操作 | `bimanual_microwave_cook` |
| 双臂 | 拍照 | `bimanual_photograph` |
| 双臂 | 解锁平板 | `bimanual_unlock_ipad` |

## 开始使用

在支持 AVX2/FMA 的 Linux x86_64 上使用参考环境：

```bash
git clone https://github.com/dexjoco-x/DexJoCo-X-public.git
cd DexJoCo-X-public
conda create -n dexjoco-x --file requirements/native-replay-linux64.conda.txt
conda env config vars set -n dexjoco-x OPENBLAS_CORETYPE=Haswell \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
conda activate dexjoco-x
python -m pip install -r requirements/native-replay-pip.txt
python -m pip install --force-reinstall --no-deps numpy==1.26.4 PyOpenGL==3.1.10
pip install -c requirements/constraints-python311.txt \
  -e './simulator[benchmark,dataset,download]'
python tools/runtime_assets.py check
```

下一步按照 [快速上手](docs/QUICKSTART.zh.md) 下载一个任务、读取示范并接入策略。下载工具的 `--dry-run` 选项可以查看所选任务的下载大小。

## 使用文档

| 想做什么 | 文档 |
|---|---|
| 安装、读取第一条示范 | [快速上手](docs/QUICKSTART.zh.md) |
| 下载数据、理解字段、转换格式 | [数据集](dataset/README.zh.md) |
| 创建和使用仿真环境 | [仿真环境](simulator/README.zh.md) |
| 接入策略、运行统一评测 | [评测接口](evaluation/README.zh.md) |
| 接入 π₀.₅ / Ego-π | [策略示例](examples/pi05/README.zh.md) |
| 用已有示范扩增场景 | [场景扩增](collection/README.zh.md) |
| 核对数据与环境版本 | [版本说明](docs/VERSIONING.zh.md) |

训练可选用自己的框架。公开内容包括数据、环境和评测工具；策略权重、模型训练实现及手套录制工具单独管理。

示范动作回放请参阅[参考环境说明](docs/REPLAY_REPAIR.zh-CN.md)。

## 联系方式

使用中有任何问题或建议，欢迎联系姜相维：[jxw@std.uestc.edu.cn](mailto:jxw@std.uestc.edu.cn)，也可以提交 [GitHub Issue](https://github.com/dexjoco-x/DexJoCo-X-public/issues)。

## 引用与许可

论文引用见 [BibTeX](CITATION.bib)。原创代码采用 [MIT](LICENSE)，原创数据贡献采用 [CC BY-NC-SA 4.0](dataset/LICENSE_SCOPE.md)。第三方模型与代码保留各自条款，详见 [来源说明](THIRD_PARTY_NOTICES.md)和[许可状态](LICENSE_STATUS.md)。
