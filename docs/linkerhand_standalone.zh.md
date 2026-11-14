# LinkerHand O6、L6、L30

[English](linkerhand_standalone.md) | **中文**

三款扩展手型都可通过 `make_env` 使用六个任务环境。默认 benchmark 仍为 **七手、六任务、2,100 条 Training v1 示范**；新增三手不计入该数据集或已公布的基线平均分。

## 创建环境

按照[主页](../README.zh.md)安装后运行：

```python
from dexjoco.benchmark6.environments import make_env

env = make_env("hammer_nail", "linkerhand_l30", seed=154187, pixels=False)
try:
    observation, info = env.reset(seed=154187)
    observation, reward, terminated, truncated, info = env.step(env.hold_action())
finally:
    env.close()
```

手型名称为 `linkerhand_o6`、`linkerhand_l6`、`linkerhand_l30`，任务名称与主页的六任务相同。双臂任务使用对应的左右手模型。设置 `pixels=True` 可获取图像。

动作由六维世界坐标系腕部增量与原生关节驱动目标组成。具体维数、顺序和范围见 `env.action_space` 与 `env.hand_metadata["sides"]`。原有七手的正式评测接口和评测协议继续对应固定 benchmark。

组合模型自动生成到 `$XDG_CACHE_HOME/dexjoco-x/`，默认是 `~/.cache/dexjoco-x/`。缓存根据资产、适配代码、MuJoCo 版本和安装位置区分，不覆盖仓库中的原七手模型。

## 回放检查示例

仓库附带 **18 段小型动作序列**，每个手–任务组合一个场景，无需下载数据集。它们通过正常执行器接口和物理接触完成任务，用于检查交互；不是训练策略的评估结果，也不属于 Training v1。

```bash
# 三款手、六个任务
MUJOCO_GL=osmesa python examples/linkerhand/replay.py --workers 4

# 单独回放 L30 锤钉，并保存视频
MUJOCO_GL=osmesa python examples/linkerhand/replay.py \
  --hand linkerhand_l30 --task hammer_nail --video outputs/l30_hammer.mp4
```

脚本检查文件哈希、任务成功信号与仿真警告；L30 锤钉还会逐次检查是否由锤头平端面接触钉帽。结果写入 `outputs/linkerhand_replay.json`，检查失败时返回非零退出码。回放保存的动作时，请使用安装说明中的数值环境。

## 模型与检查

三款手均包含可见、可碰撞的腕部连接件，保留原腕部坐标、手部质量和惯量。连接件是仿真适配件。L30 的腕部前还有约 230.5 mm 的前臂段，任务模型保留了这部分长度。

修复包括联动约束、有效驱动范围、有限力矩、接触参数，以及 L6 掌壳凸包误挡拇指的问题。仅排除对应内部连接件之间的假碰撞，仍保留它们对外部物体和其他手指的碰撞。L30 的 iPad 抓握按手掌下方的各手指分组，避免将共用腕关节误认为同一个手指。

```bash
python -m unittest discover -s tests -p 'test_linkerhand*.py' -v
python -m unittest discover -s tests -p 'test_ipad_hand_branches.py' -v
```

检查覆盖左右手、所有主动关节的运动与释放、联动、驱动力矩、外部障碍物接触、模型来源哈希、六任务入口、重置以及缓存完整性。具体数值设置见[英文说明](linkerhand_standalone.md#simulation-settings)。

## 来源

原始资产由 [h228zhou](https://github.com/h228zhou) 在 [PR #4](https://github.com/dexjoco-x/DexJoCo-X-public/pull/4) 中贡献，来自 [linker-bot/linker-sim-assets](https://github.com/linker-bot/linker-sim-assets/tree/d0224b8da58802fe7cb2e35dcc9771693c4543b6)。每款手的 `SOURCE_PROVENANCE.json` 保留原始哈希与本地修改。

该源版本未找到明确的资产许可证，不能套用 L20 的许可证；见[第三方说明](../THIRD_PARTY_NOTICES.md)。
