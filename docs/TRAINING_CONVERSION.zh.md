# 训练动作转换

[English](TRAINING_CONVERSION.md) | [中文](TRAINING_CONVERSION.zh.md)

可直接运行的完整流程见 [π₀.₅ Training v1 示例](../examples/pi05/README.zh.md)：下载 → 训练 → 策略服务 → 评估。

## 下载已验证的训练版 v1

训练版包含 **2100条重新记录的轨迹、933865帧**，每个手型与任务组合仍为50条。
在参考环境中，全部最终标签均通过连续物理回放、任务成功和状态一致性检查；
Being-H和官方LeRobot 0.4.4加载器的全量读取检查也已通过。

下载包约5.21 GB，包含配套视频、原始张量、固定统计量，以及两种预转换读取格式。
Training v1 是当前统一版本，包含同一批轨迹的 float64 动作和训练格式。请下载到新的目录；旧版仅用于历史追溯。

```bash
export DATA_ROOT="$PWD/outputs/training_v1"
python tools/dataset_packages.py download --tasks all --data-root "$DATA_ROOT"
```

包内通过TAR硬链接避免重复存储相同视频，解压后都是普通本地文件，不包含软链接。
两种训练视图分别是`prepared/beingh_final_v3/`和`prepared/lerobot_final_v3/`，
配套统计量分别为`prepared/beingh_final_v3_train_stats.json`和
`prepared/lerobot_final_v3_train_stats.json`。

在参考仿真环境中执行最终训练标签回放：

```bash
python -m pip install --no-deps -r requirements/training-replay-pip.txt
python evaluation/audit_training_replay.py --data-root "$DATA_ROOT" \
  --prepared-name lerobot_final_v3 --workers 4 --output training_replay.json
```

包内`metadata/VERIFICATION.json`和逐条报告记录了验证结果。验证执行保存的示范标签，
不代表训练后模型的成功率；本次修复没有降低成功标准，也没有改动环境模型。

官方LeRobot的读取示例见[英文版](TRAINING_CONVERSION.md#download-verified-training-v1)。
该包使用native200动作表征，标签已经归一化，训练时应关闭额外的状态/动作归一化，
同时正确处理有效维度和时间填充掩码。推理时使用下文的配套适配器与统计量，
将标签还原为float64实际指令，并保留该精度发送到环境。已有模型继续使用训练时的
统计量与解码器，不能通过直接更换解码器来修复已有模型。

旧转换按数据分位数裁掉了部分有效示范动作，例如把 1.61 厘米的腕部移动压成
0.41 厘米。新的 `quantile_affine_v3` 只缩放，不删除中央 98% 范围以外的值。
标签可以超过 [-1, 1]；实际控制器的物理限制继续保留。

v3 另外使用 2 的整数次幂缩放，避免几乎不变的通道产生极大的训练标签。解码时先撤销这次缩放，
再进行逆变换。对已经采集的 v2 标签，`rescale_training_bundle.py` 逐值检查解码指令完全相同，
并保留对应的视频和状态；缩放系数也会兼顾极小指令，避免舍入为零而改变执行值。

统计量中记录归一化版本，新导出使用独立目录。无版本标记的旧统计量继续使用
`quantile_clip_v1`，保证旧检查点仍按原规则解码。不要直接给已有检查点更换统计量或解码器。

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT"
python tools/dataset_smoke.py --data-root "$DATA_ROOT" --output data_check.json
```

输出目录为 `prepared/beingh05_affine_v3/`，统计量、数据清单和训练配置使用同一前缀。
对于训练版v1，转换器自动读取 `bundle/training_statistics.json`。
原始数据的转换实验需要显式添加 `--allow-unverified-conversion`，之后仍需独立回放，
并确保观测与实际执行一致，才能作为已验证的训练数据。
需要另一次导出时使用 `--output-name NAME`；重新采集后使用
`--statistics-file FILE` 复用已经冻结的统计量。统计量应在最终标签采集前固定。

新解码器读取 float32 标签后，使用 float64 计算。分位数相等的通道采用
`标签 = 原值 - q01`，保留少数帧中的动作。在实际执行器边界处向合法范围内舍入，
避免解码后又被控制器裁掉。这个边界处理与按数据分位数裁剪不同。

数值检查之后还要检查真实任务。在[参考环境](REPLAY_REPAIR.zh-CN.md)中安装额外的
Parquet 读取依赖，再直接回放导出的文件：

```bash
python -m pip install --no-deps -r requirements/training-replay-pip.txt
python evaluation/audit_training_replay.py --data-root "$DATA_ROOT" \
  --prepared-name beingh05_affine_v3 --workers 4 --output training_replay.json
```

检查分别报告任务是否完成、记录状态是否一致。如果转换后的指令改变了轨迹，就要从实际执行
重新生成 RGB 和状态。重新采集时应执行最终标签解码出来的动作，保存同一份标签，再从保存的
训练文件独立回放。仅完成转换，不能据此宣布 Raw v1.1 已成为通过验证的 float32 训练版本。

两个读取器需要不同的路径模板。默认导出保留 Being-H 使用的按轨迹路径；
`--reader lerobot` 导出官方 LeRobot v3 使用的文件索引路径。两份导出保存相同的标签，视频共享。
在独立的读取环境中安装 `lerobot==0.4.4` 后检查：

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT" --reader lerobot \
  --statistics-file "$DATA_ROOT/prepared/beingh05_affine_v3_train_stats.json"
python tools/check_lerobot.py --data-root "$DATA_ROOT" \
  --prepared-name lerobot_affine_v3 --output lerobot_reader_check.json
```

该检查核对全部数值标签、抽取的视频帧和轨迹末尾的动作块。LeRobot 会重复末帧值来补齐动作块，
训练时须同时使用 `action_is_pad` 和 `action.valid_mask` 排除补齐项；Being-H 适配器则将这些项置零。
保存的标签已经归一化，应使用绑定的解码器，不要再做一次归一化。

使用新标签训练的 native200 策略，在 `arm40_v1` 推理服务端使用对应适配器：

```python
from dataset.tools.policy_adapter import Native200PolicyAdapter

adapter = Native200PolicyAdapter(training_statistics['groups'])
state, state_mask = adapter.state(observation)
# prediction 为按这些标签训练的模型输出，形状为 float32 [horizon, 200]
reply = {'actions': adapter.actions(prediction, observation)}
```

返回的 float64 指令应直接发送，不要再转回 float32。Training v1 的 π₀.₅ 示例
在拆分 32 维 token 后使用这一解码器；旧检查点保留原有统计量和解码方式。

重新采集的 `native.npz` 另外保存实际执行过的 `training_labels`。导出工具和便携读取器都原样保留
这些标签，并检查统计量是否匹配。RGB 和状态来自同一次执行；这是新采集的训练轨迹，不是把原有
float64 轨迹无损转换成 float32。

`collection/collect_training_labels.py` 可以在执行最终标签的同时跟随原示范的腕部目标。
锤钉采集还可通过 `--hammer-strokes` 显式增加抬锤和敲击步骤。这些步骤属于新示范，
会保存为动作并生成对应 RGB 和状态。采集参数会被记录，结果要独立回放；评估环境不会额外补动作。

这里执行的是保存的标签，不是训练后的模型预测；回放结果与策略学习效果应分别报告。
