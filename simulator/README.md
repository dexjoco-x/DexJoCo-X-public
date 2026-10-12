# 仿真环境

MuJoCo 环境覆盖七种灵巧手和六个任务，机器人、物体模型及纹理随包提供。

## 创建环境

完成 [安装](../README.md#开始使用) 后，在 Python 中运行：

```python
from dexjoco.benchmark6.environments import make_env

env = make_env("pick_bucket", "allegro", seed=0, pixels=False)
try:
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(env.hold_action())
finally:
    env.close()
```

`make_env(task, hand, seed=...)` 接受首页列出的任务和手型。`pixels=True` 启用图像观察；`hold_action()` 返回保持当前姿态的动作，方便检查环境。

## 控制与观察

腕部动作使用世界坐标系下的位置和旋转增量，手部动作使用执行器目标值。执行器名称、范围和成功条件记录在 [任务合约](../evaluation/contracts) 中。

统一策略接口使用图像、状态、掩码和任务指令，见 [评测文档](../evaluation/README.md)。相机设置见 [相机配置](../dataset/CAMERA_PROFILE.md)。

## 相关模块

- `dexjoco.benchmark6.environments`：环境入口。
- `dexjoco.benchmark6.success_policy`：成功判定。
- `dexjoco.benchmark6.dataset`：示范数据接口。
- `dexjoco.autocollect`：已有示范的场景扩增。

`teleoperation/` 保存环境引用的模型资产。第三方来源与条款见 [来源说明](../THIRD_PARTY_NOTICES.md)，兼容标识见 [版本说明](../docs/VERSIONING.md)。
