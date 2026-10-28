# 示范动作回放

[English](REPLAY_REPAIR.md) | [中文](REPLAY_REPAIR.zh-CN.md)

v1.1 的 2,100 条示范均在参考环境下通过连续动作回放和逐步状态核对。
旧版仍可通过 revision `9f48e1404168227a22586af0cffb8371dbb7e1f1` 获取。

## 参考环境

需要支持 AVX2/FMA 的 Linux x86_64 与 Conda。请创建独立环境：

```bash
conda create -n dexjoco-replay --file requirements/native-replay-linux64.conda.txt
conda env config vars set -n dexjoco-replay OPENBLAS_CORETYPE=Haswell \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
conda activate dexjoco-replay
python -m pip install -r requirements/native-replay-pip.txt
python -m pip install --force-reinstall --no-deps numpy==1.26.4 PyOpenGL==3.1.10
```

相同的 MuJoCo 版本号可能对应不同的二进制构建。这里固定 Conda-forge 构建，
使用 NumPy wheel，并由启动器在导入数值库前固定 BLAS CPU 内核与线程数。

## 检查保存的动作

```bash
python tools/replay_reference.py /path/to/dataset --workers 4 --output /tmp/native-replay.json
```

数据目录包含 `bundle/manifests/`，也可以直接传入 `bundle/`。
默认检查全部清单，包括训练示范；可以用 `--task`、`--hand`、`--scene` 缩小范围。
运行时逐条保存结果，所有选定示范均成功才返回退出码 0。

加上 `--check-states` 可同时要求每一步保存的 TCP 位姿和手部关节状态与回放的误差不超过
`2e-5`。FK 在独立渲染数据上计算，不重置或改写物理状态。

检查包括 tensor 校验和、初始机器人状态与模型安装信息。随后连续执行保存的动作，
遇到终止、截断或动作耗尽即停止；成功必须在该序列中真实触发。

## 模型修正

XHand 拍照与解锁 iPad 的示范需要修正后的左手安装模型。现在除了版本标识，
也检查实际安装朝向。两组原有动作在独立参考环境下均为 50/50 回放成功。
XHand 微波炉模型此前已采用正确安装方式。

本版有 220 条动作序列重新进行了物理采集；另有 918 条保留原动作前缀，移除了任务终止后
不再执行的尾段。全部 2,100 条示范在参考环境中同步导出观测，共 939,356 帧。
这些检查针对 v1.1，不代表旧版回放结果被重新标为成功。

## 数据验证与动作精度

验证器会比较保存的控制约定与实际运行环境，并检查数组形状、有效通道、相机掩码、时间戳、填充区、四元数和动作范围。文件校验值匹配不代表内容本身正确。

v1.1 的原始动作回放应直接读取 native.npz 并保留保存的精度。全量 2,100 条动作转为 float32、经正常策略接口执行后，成功 1,707 条（81.3%），执行报错 0 条；原始保存精度下则为 2,100/2,100 成功。[逐条结果](replay/float32_precision_2100.json)记录了此次精度诊断。这不是学习策略成功率，也不能据此声称 v1.1 已兼容 float32 精确回放。

新的 float32 采集可使用 collection.action_precision.float32_native_action，将边界向内舍入为可准确保存的数值。应先转换，再执行并保存同一指令，随后进行独立回放和同步图像导出。把此函数直接应用于旧数据不等于完成验证；新采集试验与公开 v1.1 数据分开保存。

在参考环境中运行精度检查：

```bash
python evaluation/audit_action_precision.py /path/to/dataset \
  --task hammer_nail --hand allegro --scene scene_049 \
  --output /tmp/float32-probe.json
```

不加任务、手型和场景筛选时会检查全量数据。此命令将保存动作转成 float32，再通过正常策略接口及其已有的动作限幅执行；只要选中序列有失败，退出码即非零。它不会追加动作、逐步恢复仿真状态或修改数据文件。最初用户报告使用的是 float64；float32 敏感性是后续检查另行发现的问题。
