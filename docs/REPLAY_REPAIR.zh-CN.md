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
