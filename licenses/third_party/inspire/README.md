# Inspire 模型文件的许可与署名

来源：Dex-URDF，作者署名保留为 dex_urdf authors；上游版权标注为 Copyright 2024。
模型基于 Inspire 官方 STEP，经过 Dex-URDF 作者修改。原始说明完整保留在 `UPSTREAM_NOTICE.txt`。

上游固定版本为 `f5e7132f22108164577fea4c25ef99b5cc0e1900`，许可声明：https://github.com/dexsuite/dex-urdf/blob/f5e7132f22108164577fea4c25ef99b5cc0e1900/README.md#robot-source 。
CC BY-NC-SA 4.0 全文保留在 `CC-BY-NC-SA-4.0.txt`，官方链接：https://creativecommons.org/licenses/by-nc-sa/4.0/ 。

- `VERIFIED_FILES.json`：8 个右手碰撞网格，文件内容与上游完全一致，仅改变文件名和存放路径。
- `DERIVATION_SOURCES.json`：9 个视觉 GLB 原文件的上游 Git blob ID、SHA-256 与固定版本链接。
- `DERIVED_FILES.json`：68 个转换/镜像/材质拆分网格的逐文件证据与修改说明。双向顶点距离不超过 20 纳米，完整三角形多重集一致；镜像 OBJ 另通过逐字节转换配方验证。

复核命令（需 trimesh、NumPy、SciPy 和准确上游 GLB）：

```bash
python tools/verify_inspire_derivatives.py --upstream-dir /path/to/pinned/glbs
```

该文件集合允许在遵守原许可的情况下进行非商业分享，须保留署名、来源、许可与修改说明。所列改编网格的项目修改部分也按 CC BY-NC-SA 4.0 提供。不得暗示上游为本项目背书，许可不提供质量保证。

该声明只覆盖所列 76 个网格，不覆盖剩余 URDF 或整个项目/数据集。原始碰撞网格、右视觉 OBJ、镜像左模型和材质拆分副本的几何均未在此次核验中修改。
