# DexJoCo-X 数据字段与转换

原始数据版本为 `dataset-v1.0-corrected`：七手、六任务、2,100 条示范和 946,156 帧，20 Hz；每个手–任务组合 50 条，全部属于训练集。闭环评测采用独立场景种子。下载步骤见 [数据指南](DOWNLOAD.md)。

## 原生记录

`bundle/manifests/<task>.json` 索引每条示范的 hand、scene_id、seed、steps、split、张量、合约和视频路径，并绑定文件 SHA-256。

`native.npz` 含 state、action、state_mask、action_mask、arm_mask、timestamp 和 image_mask。双臂数组的侧序为右、左；缺失侧和填充维度由掩码标识。每侧状态前七维为腕部世界位置与 wxyz 四元数，后面为原生手状态；动作前六维为世界坐标腕部位移/旋转增量，后面为 actuator 目标值。腕部位置单位为米，旋转单位为弧度；手通道名称、顺序和范围见该 episode 的合约。

视频为 256×256 RGB、20 Hz 的 H.264，视图包含 front、wrist_right，以及双臂任务的 wrist_left。有效长度使用 manifest 的 steps。任务指令和成功条件由对应合约记录。

## 转换视图

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT"
python dataset/tools/prepare_benchmark7_repr.py "$DATA_ROOT" --representation faas \
  --contracts representations/table1_sided_v3/codecs/faas7_project_v3_sided.json
```

Native 输出至 `prepared/beingh05/`，FAAS 输出至 `prepared/beingh05_faas/`。目录名称保持历史兼容，它们是通用 Parquet/视频数据，不包含模型训练实现。两者的 state/action 宽度为 200：右侧占 0:40，左侧占 40:80，其余为零；每侧腕部占前六维，状态旋转使用 rotvec，后面为手坐标。Native 保留原生手坐标，FAAS 使用 32 个功能槽位。

Parquet 字段包括 observation.state、observation.state_mask、action、action.valid_mask、timestamp、frame_index、episode_index、index 和 task_index。meta 中提供任务指令、episode 索引、原始训练统计量及格式信息。每个手–任务组分别计算 q01/q99，转换后的值已归一化并裁剪到 [-1, 1]，无效维度归零；对接训练框架时避免重复归一化。

`tools/dataset_smoke.py` 检查已安装任务的原始数组、转换数值、掩码、episode 索引和首尾视频帧。转换视频优先使用硬链接，跨文件系统时复制，不创建软链接。

DexLatent 视图另需兼容的本地编码器及匹配记录，见 [转换说明](DOWNLOAD.md)。下载包不含任何预转换缓存或编码器权重。π0.5 示例采用的 Ego-π 归一化与这里的有界转换不同，训练和推理必须使用相同的统计量和规则，见 [推理示例](../examples/pi05/README.md)。

XHand 微波炉的 50 条左手轨迹已按 `xhand_left_mount_rx180_v1` 修正。论文历史模型和修正数据的关系见 [版本说明](../docs/VERSIONING.md)。
