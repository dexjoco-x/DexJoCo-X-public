# 相机配置

默认配置为 `dexjoco_official_allegro60_hanoi_front_v1`。通过 TCP 的 `attachment_site` 坐标系，将 Allegro 的官方相机外参应用到七种手型。

| 设置 | 配置 |
|---|---|
| 图像 | RGB，256×256，20 Hz，动作执行前采样 |
| 常规任务视场角 | 60° |
| 汉诺塔 front 相机 | 世界位置 [1.3, 0, 1.6] m，视场角 45° |
| 视频有效长度 | 对应 manifest 的 `steps` |

当前仿真环境默认启用该配置。使用旧版本相机训练的策略时，在导入仿真模块前设置 `BENCHMARK7_CAMERA_PROFILE=common_tcp_side85_v1`，使评测观察与训练保持一致。

旧数据的相机转换和数值回放记录见 [历史配置](../docs/maintenance/CAMERA_PROFILE_HISTORY.md)。
