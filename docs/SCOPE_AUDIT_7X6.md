# 七手六任务发布范围审计

> 2026-10-05 更新：评测代码与干净数据仓库已获负责人公开授权。本文其余可见性描述为原验收时的历史状态；权重、手套映射及完整归档继续私有。

审计日期：2026-10-01。本文区分论文正式范围、发布候选中可恢复的缺失文件，以及开发工作区残留。删除残留之前必须以最终 42 个评测合约为入口重新运行依赖与冒烟检查。

## 正式范围

- 手：`allegro`、`inspire`、`leap`、`linkerhand_l20`、`sharpa_wave`、`wuji`、`xhand`。
- 任务：`hammer_nail`、`pick_bucket`、`single_hand_hanoi`、`bimanual_microwave_cook`、`bimanual_photograph`、`bimanual_unlock_ipad`。
- 数据：42 个 hand–task 组合，每个组合 50 条，共 2,100 条。
- 策略动作表征：Native、FAAS、DexLatent。

`evaluation/contracts/` 恰好包含上述 7 × 6 个合约；`dataset/catalog.json` 恰好包含上述六个任务。

## 策略动作表征

Table I 使用的 `representations/table1_sided_v3/` 是完整的冻结包。FAAS 合约中的手键恰好是正式七手，DexLatent 的 `selected.pt` 与冻结 seal 均已保留。`evaluation/codecs/faas7_project_v1.json` 是旧副本，公开版应只保留或明确指向 Table I 的 sided-v3 合约，避免两个版本并列造成误用。

## 人工采集映射

用户于 2026-10-01 决定全部手套映射不公开。发布候选已移出手套映射实现、校准配置、实时输入桥接及专用测试，私有工作区保留完整备份。XHand 映射已从实际录制脚本调用的 `preview_xhand_rokoko.py` 恢复后归入私有备份；LinkerHand 实时手套入口尚未找到，但不再影响本次发布范围。

冻结 XML 所需的 Inspire、Wuji 和 Sharpa 模型资产保留在历史路径中。静态关节命名、执行器顺序与初始姿态从映射目录解耦到 `dexjoco.sim.envs.model_profiles`，这些是仿真模型定义，不包含手套到机器人的映射算法。

原完整工作仓库的历史曾包含映射，继续保持私有。2026-10-02 发布仓库已同步独立的干净历史，旧分支版本和标签作本地镜像备份。当前发布历史不含手套映射、训练实现或权重；完整工作仓库未改动。

## 任务与环境收口

公开 `tasks.CONFIG_MAPPING` 由七手六任务生成，恰好包含 42 个 `task/hand` 键；正式环境类表同样只包含六任务。OYMotion、Shadow、OpenArm、UR10e/Rizon 变体，以及浇花、鼠标、眼镜、夹钳、装配等旧任务实现和资产已移出发布源码，保存在本地私有范围备份中。

单臂汉诺塔保留共享的底层 Hanoi 逻辑和必要模板，但没有双臂汉诺塔的公开注册项。8 个原生动作示范读取入口单独保留在 `tasks/legacy_sources.py`；它们继续使用原动作维度、控制器及增量限幅，不能通过名称替换冒充正式评测环境。人工遥操作包装器和旧绝对位姿 OpenPI 客户端已归入私有备份，公开客户端使用 `evaluation/runtime.py`。

源模型和录制接口使用的仿真关节定义保留原数值。冻结正式模型及数据不重新生成；来源清单改为读取当前安装包的实际模块，并正确解析保留的模型来源证据。所有 42 个模型的资产指纹验证均通过。

## 清理后的验证含义

“重新运行采集”不表示重新采集 2,100 条数据。验证分为三层：

1. `collection/smoke.py`：创建一个正式环境、构造观测和动作、执行一步并关闭环境；
2. 发布范围检查：确认发布源码与 wheel 均不含手套映射、校准配置或实时桥接；
3. 每个任务执行一次临时短轨迹导出与回放，检查 RGB/state/action/mask 对齐，不写入正式数据集。

完成范围清理后，还需重跑 42/42 环境可移植性测试、三种动作表征 smoke、数据 manifest 校验和发布静态审计。

本次白名单与依赖修复后，42/42 正式环境的初始化、相机渲染、Native/FAAS/DexLatent 编解码及四步物理检查全部通过；采集接口的一步 smoke 和 2,100 条 manifest 静态校验也通过。本次实际裁剪后的结果以 `SCOPE_SMOKE.json`、`SOURCE_COMPATIBILITY_SMOKE.json` 及重新生成的环境/表征 smoke 为准。

## 移出手套映射后的验证

当前源码通过 42/42 正式环境初始化与一步物理执行、Native/FAAS/DexLatent 运行时检查、采集接口检查，以及 LinkerHand/XHand 自动扩增源与目标 FK 初始化。iPad 点击序列从历史模块解耦后已检查正确数字推进，源模型生成器的冻结文本保留用于原有哈希验证。模型资产清单已按移出映射后的内容重新生成。

独立解包安装后的 wheel 同样通过 42/42 环境和自动扩增源/目标 FK 初始化；wheel 中手套映射、校准配置与桥接文件数量为零。结果见 `GLOVE_RELEASE_SCOPE_SMOKE.json`。

## 最终范围裁剪验证

裁剪后的源码与独立 wheel 均通过 42/42 正式环境验证；Native/FAAS/DexLatent 运行时检查、六任务两帧真实 RGB 保存与确定性回放、8 个原示范接口的单步执行及 wheel 初始化均通过。50 个正式/示范接口都拒绝人工录制模式。12 个被审计资产目录的哈希验证通过，2,100 条数据清单与冻结正式模型未改动。

被移出的历史实现和资产保留在本地私有范围备份中，GitHub 仍保持私有。剩余公开条件为许可证、LinkerHand 型号、最终作者/引用信息、数据版本发布决定及干净 Git 历史。模型权重按用户决定保持私有。
