# 示范回放

[English](REPLAY_REPAIR.md) | **中文**

训练和回放统一使用 **Training v1**。[下载当前数据](../dataset/README.zh.md)。
2,100 条最终训练标签均在参考环境下完成任务，并通过逐步状态核对；包内 float64
动作与这些标签解码后的指令完全一致，三种格式的视频相同。

## 参考环境

回放 Training v1 前请先更新代码。此前控制器中的反三角函数会随 CPU 选择不同的
实现，因此即使依赖版本和 BLAS 检查通过，轨迹仍可能出现偏差。当前控制器使用
计算顺序固定的标量旋转计算，检查报告同时记录 NumPy 的 CPU 运算路径、系统库
哈希和控制器数学实现版本。这次修改代码，不替换 Training v1 数据包或视频，
任务成功条件和 `2e-5` 状态检查阈值均保持原值。
修复后，关闭 NumPy AVX-512 的 2,100 条原生动作回放全部通过；Ubuntu 22.04 中
覆盖全部手–任务组合的 84 条复测也全部通过。[验证摘要](replay/controller_math_v1.json)。

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

## 检查 LeRobot 中保存的标签

```bash
python -m pip install --no-deps -r requirements/training-replay-pip.txt
python evaluation/audit_training_replay.py --data-root "$DATA_ROOT" \
  --prepared-name lerobot_final_v3 --workers 4 --output training_replay.json
```

标签为 float32；使用包内统计量解码为 float64 后执行。完整说明见[训练使用](TRAINING_CONVERSION.zh.md)。
这些检查衡量示范回放，不是训练后策略的成功率。旧版精度诊断和修复记录已归入
[版本历史](VERSIONING.zh.md)，不代表当前 Training v1 的检查结果。
