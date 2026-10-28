# 场景扩增

[English](README.md) | **中文**

基于已有机器人示范，在随机场景中执行任务，验证成功后导出新的示范。

流程：**读取示范 → 采样场景 → 执行动作 → 成功判定 → 导出数据**。

## 组件

| 模块 | 用途 |
|---|---|
| `dexjoco.benchmark6.environments.make_env` | 创建任务与手型环境 |
| `dexjoco.autocollect` | 场景扩增与执行 |
| `dexjoco.benchmark6.success_policy` | 任务成功判定 |
| `dexjoco.benchmark6.export.export_episode` | 导出图像、状态、动作和掩码 |

运行环境连通性检查：

```bash
python collection/smoke.py
```

该命令创建环境并执行一步。人工手套映射、校准和实时录制工具单独管理；本模块从已有机器人示范开始。

此 smoke 测试只检查一次仿真步进，不代表 42 个组合的完整采集流程已验证。部分试验性实时观测适配器尚未实现。旧 export_episode 入口需要采集源中的 replay.zarr、privileged.zarr 和 final_truth.json；公开训练 native.npz 本身不提供这些输入。
