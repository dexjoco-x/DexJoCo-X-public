# DexJoCo-X 仿真运行包

本包提供七种灵巧手、六个任务的冻结 MuJoCo 环境、统一数据接口与自动场景扩增。人工手套映射和录制工具不公开。

## 安装

使用 Python 3.10 或 3.11，从仓库根目录执行：

```bash
pip install -e './simulator[benchmark]'
python collection/smoke.py
python tools/portability_smoke.py
python tools/representation_smoke.py --representations native faas
```

无显示器环境中执行真实 RGB 和诊断回放检查，需要系统安装 OSMesa：

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python tools/scope_smoke.py
```

## 正式环境

手型为 Allegro、Inspire、LEAP、LinkerHand、Sharpa Wave、Wuji 和 XHand。历史内部键 `linkerhand_l20` 保持兼容，硬件型号说明见发布门槛文档。

任务为 `hammer_nail`、`pick_bucket`、`single_hand_hanoi`、`bimanual_microwave_cook`、`bimanual_photograph` 和 `bimanual_unlock_ipad`。

```python
from dexjoco.benchmark6.environments import make_env

env = make_env("pick_bucket", "allegro", seed=0, pixels=False)
try:
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(env.hold_action())
finally:
    env.close()
```

`dexjoco.tasks.CONFIG_MAPPING` 恰好包含 42 个 `task/hand` 键。正式动作使用 TCP 世界系增量及原生执行器目标；不要与历史绝对位姿接口混用。统一策略推理和评测入口在仓库的 `evaluation/` 中。

## 数据与场景扩增

`dexjoco.benchmark6.dataset` 负责带掩码的 state/action 数据；`dexjoco.autocollect` 和 `dexjoco.benchmark6` 提供已有示范的扩增、成功验证与回放导出。`tasks/legacy_sources.py` 单独保留 8 个冻结示范读取入口，沿用原控制器及限幅，不新增基准任务，也不提供人工录制。

`teleoperation/` 仅保留冻结 XML 所需模型资产，资产授权状态见仓库的 `docs/ASSET_LICENSE_AUDIT.md`。旧任务、额外手型和旧绝对位姿 OpenPI 客户端均不属于当前发布范围。

DexLatent 编码器权重保持私有；使用该表征时需额外安装 `./simulator[dexlatent]` 并准备匹配的本地编码器。无显示器时先同时设置 `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`。
