# 仿真资产随包安装验收

2026 年 10 月 2 日，已构建 `dexjoco_x-0.1.0-py3-none-any.whl`，大小 102,380,632 字节（约 102 MB）。源码仓库包含实际资产文件，wheel 同时包含正式环境所需资产与原许可记录。

本次将 wheel 安装到独立目标目录，在禁用源码导入路径后验证：

- 801 个必要资产全部存在，SHA-256 与冻结清单一致，共 252,047,145 字节，无遗漏或改动。
- 18 个第三方许可与来源记录文件完整保留；根目录原创代码许可、许可范围和第三方说明也随包保留。
- 七手六任务全部 42 个环境均能初始化、重置并执行物理步进，关节状态有限。
- wheel 未包含策略/编码器权重及手套映射 Python 代码。

以上检查为安装与运行检查，没有重新测量策略任务成功率，也不改变第三方资产的许可核验状态。机器可读结果见 [BUNDLED_ASSET_VALIDATION.json](BUNDLED_ASSET_VALIDATION.json)。

## 从本仓库构建

使用 Python 3.11，在仓库根目录执行：

```bash
python -m pip wheel --no-deps ./simulator --wheel-dir ./dist
```

生成的 wheel 可直接交给使用者安装：

```bash
python -m pip install -c requirements/constraints-python311.txt \
  './dist/dexjoco_x-0.1.0-py3-none-any.whl[benchmark,dataset,download]'
```

系统图形库与其他 Python 依赖仍按 [快速上手](QUICKSTART.md) 安装。上述安装不需要从其他项目下载机器人模型或任务素材。完整示范数据通过本项目的 Hugging Face 下载入口提供。

wheel SHA-256：`b620a99ac5e5cde450f50ca56e0d2c51275c00e573a48fca74f4bb185c41c3a1`。
