# DexLatent / XHand 来源记录

本项目 XHand 原始模型来自官方 [DexLatent 仓库](https://github.com/EmptyBlueBox/DexLatent)，固定版本为 `502750df34b9fa6a1b56a66481586887ec722f9c`。

该版本的根目录 MIT 许可证署名为 Copyright (c) 2026 Yutong Liang，全文保存在本目录 `LICENSE`。其内容与此前本地 `xhand/SOURCE_LICENSE` 一致。固定版本树中未发现另行约束 XHand 文件的子目录许可证。

`VERIFIED_FILES.json` 列出的 62 个文件与该版本 `Assets/xhand/` 中的文件逐字节一致，包括 60 个正式评测环境引用的网格以及左右手源 URDF。只调整存放路径，保留了许可证、版权声明与上游链接。

MJCF 转换、固定链接惯性合并等项目修改记录保留在 `xhand/left/metadata.json`、`xhand/right/metadata.json` 与 `simulator/scripts/import_xhand.py`。本记录不把这些生成文件宣称为上游原文件，也不替本项目原创代码指定许可证。

这里只记录官方发布版本的来源及其声明的许可，不对其他 XHand 版本、硬件、商标或专利作授权声明。
