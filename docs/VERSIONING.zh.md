# 当前标准与历史版本

[English](VERSIONING.md) | **中文**

## 当前统一标准

使用 **Training v1**（`training-float32-v1`）、当前代码和[参考环境](REPLAY_REPAIR.zh-CN.md)。
当前指南中的下载命令均指向这一版：7 种手、6 个任务、2,100 条示范、933,865 帧。
其中 float64、LeRobot 和 Being-H 文件对应同一批实际执行的轨迹。[下载数据](../dataset/README.zh.md)。

训练预算为每个手–任务组合 50 条示范；策略评估为每个组合独立运行三次、每次 50 条，
使用[评估指南](../evaluation/README.zh.md)规定的场景种子和成功条件。
报告结果时同时记录数据版本、代码提交、模型检查点和评估设置。

Training v1 使用现有参考物理环境和已修复的模型。float32 标签配套
`quantile_affine_v3` 统计量与 float64 解码器。本次统一下载入口没有改变归档、轨迹或成功判据。

## 历史记录

旧版仅用于追溯已有结果，不再作为新实验的并列选项，也不再放在默认数据目录中。

| 历史快照 | 用途 |
|---|---|
| [Raw v1.1](https://huggingface.co/datasets/darenrenjian/DexJoCo-X/tree/dataset-v1.1-native-replay) | 早期 float64 数据和回放调查 |
| [最初公开快照](https://huggingface.co/datasets/darenrenjian/DexJoCo-X/tree/dataset-v1.0-corrected) | 复现早期公开下载 |
| `paper-v1` | 论文原实验和模型检查点 |

论文原有结果使用早期数据，没有改记为 Training v1 的结果。旧模型仍使用各自训练时的统计量
与解码器。Training v1 和 Raw v1.1 共享场景编号，合并它们不会增加独立场景。
[Issue 历史](https://github.com/dexjoco-x/DexJoCo-X-public/issues/1)保留了修复经过。

`linkerhand_l20` 是保留的文件和合约键名；源几何基于官方 L25，公开文字称为 LinkerHand。
