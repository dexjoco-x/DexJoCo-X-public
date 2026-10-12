# 版本与兼容性

[English](VERSIONING.md) | **中文**

| 版本 | 内容 |
|---|---|
| `paper-v1` | 论文 Table I 的训练数据、checkpoint 和评测结果 |
| `dataset-v1.0-corrected` | 当前公开的 2,100 条示范 |
| `runtime-v9` | 与当前数据配套的仿真环境 |

公开数据更新了 XHand 微波炉的 50 条示范；论文 checkpoint 的训练早于该更新，未在当前公开数据上重新训练。

`linkerhand_l20` 是文件和合约中的兼容键。其左右手源 URDF 由官方 L25 文件修改，文档统一使用 `LinkerHand`；当前版本保留键名和模型几何。
