# 自动化采集与场景扩增

发布版公开已有示范的自动场景扩增、成功验证与轨迹导出，不公开任何手套映射、校准配置或实时输入桥接。人工手套录制工具保留在项目私有工作区。

公开组件包括：

- `dexjoco.benchmark6.environments.make_env`：七手六任务环境入口。
- `dexjoco.autocollect`：基于已有示范的场景扩增和执行。
- `dexjoco.benchmark6.success_policy`：统一成功条件。
- `dexjoco.benchmark6.export.export_episode`：回放验证及 RGB/state/action/mask 导出。
- `dexjoco.benchmark6.dataset`：统一 Native 数据接口。

```bash
pip install -e ./simulator
python collection/smoke.py
```

流程为：读取已有示范，绑定冻结任务与手模型，在指定随机场景执行，按任务成功条件和回放检查筛选，再导出对齐数据。用户也可提供符合接口的自有示范。此发布包不能单独完成从 Rokoko 手套输入到新人工示范的录制。

`collection/smoke.py` 创建一个正式环境并执行一步；不会重新采集或覆盖已整理的 2,100 条数据。模型资产与静态关节定义仍保留，自动扩增重定向器只处理已有机器人示范，与未公开的人类手套映射不同。

发布版公开任务使用 `task/hand` 标识。`tasks/legacy_sources.py` 中的 8 个旧标识仅用于已有示范读取，保持原动作语义；这些入口不提供人工录制。`tools/scope_smoke.py` 为六个任务分别保存两帧诊断数据和真实 RGB，再从相同种子重放以核对状态，仅写入临时目录，输出不具备训练资格。
