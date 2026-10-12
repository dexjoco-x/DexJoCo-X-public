# dexjoco

该软件包包含 DexJoCo 的核心运行时，包括 MuJoCo 仿真环境、任务配置类、演示录制与回放工具，以及用于策略 rollout 的 OpenPI 评估客户端。

软件包结构：

| 路径                     | 描述                                                                       |
| ------------------------ | --------------------------------------------------------------------------------- |
| `dexjoco/tasks/`         | 任务配置类和特定任务的环境设置。                   |
| `dexjoco/sim/`           | MuJoCo 仿真包装器、控制器和 XML 资源。                          |
| `dexjoco/data/`          | Zarr episode 存储、视频写入、深度捕获和回放支持工具。 |
| `dexjoco_openpi_client/` | 用于 DexJoCo 策略 rollout 的 OpenPI 评估客户端。                              |

## 安装

在仓库根目录下运行：

```bash
conda env create -f environment-dexjoco.yaml
conda activate dexjoco
```

### 测试安装

在仓库根目录下运行交互式环境测试：

```bash
python scripts/test_envs.py
```

使用 EGL 运行无头渲染测试：

```bash
python scripts/test_headless_envs.py
```

这两个脚本都会遍历已注册的 DexJoCo 任务配置，并使用零动作逐步执行每个环境。无头测试会将渲染后的视频写入 `test_headless_videos/`。

已注册的任务名称：

```text
bimanual_assembly
bimanual_hanoi
bimanual_microwave_cook
bimanual_photograph
bimanual_unlock_ipad
click_mouse
fold_glasses
hammer_nail
pick_bucket
pinch_tongs
water_plant
```

## 探索环境

使用顶层演示采集工具进行交互式遥操作数据采集：

```bash
python scripts/record_demos_zarr.py --exp_name water_plant
```

`--exp_name` 的值必须匹配某个已注册的任务名称。交互式采集会使用 MuJoCo viewer 和该任务的遥操作包装器。

已录制的演示可以通过策略接口进行回放：

```bash
python scripts/replay_demos_zarr.py \
  --exp_name=water_plant \
  --input_dir=./demos \
  --out_dir=./replay_output \
  --randomize=True \
  --restore_state=True
```

回放会读取包含 `replay.zarr` 的输入文件夹；如果可用，会恢复录制时的初始场景状态；随后执行保存的动作序列，并将回放得到的 Zarr episodes 和视频写入输出目录。使用 `--randomize=True` 时，回放可以基于预设摄像机、光照和纹理随机化生成 `rand_full` 视觉变体。

常用回放选项：

| 选项            | 默认值           | 描述                                                                             |
| ----------------- | ----------------- | --------------------------------------------------------------------------------------- |
| `--exp_name`      | `water_plant`     | 选择用于回放演示的任务。                                         |
| `--input_dir`     | `./`              | 包含已录制演示文件夹的目录，这些文件夹中应有 `replay.zarr`。                          |
| `--out_dir`       | `./replay_output` | 回放后的 Zarr episodes 和视频的输出目录。                                 |
| `--randomize`     | `True`            | 启用回放时的 `rand_full` 视觉随机化。                                   |
| `--restore_state` | `True`            | 当可用时，从已录制的 `state[0]` 恢复初始桌面高度和物体姿态。 |
| `--save_failed`   | `False`           | 即使环境未报告成功，也保存回放输出。                  |
| `--save_depth`    | `False`           | 在 RGB 视频之外，同时保存深度数组和深度视频。                               |

## 无头模式

为离屏渲染设置 EGL：

```bash
export MUJOCO_GL=egl
```

使用 `policy_mode=True` 和 `render_mode="rgb_array"` 构建无头环境：

```python
TaskConfig.get_environment(policy_mode=True, render_mode="rgb_array", ...)
```

常用环境构建选项：

| 选项                    | 描述                                                              |
| ------------------------- | ------------------------------------------------------------------------ |
| `policy_mode=True`        | 使用策略动作接口，并禁用交互式遥操作。 |
| `render_mode="rgb_array"` | 启用用于自动化评估的离屏图像观测。           |
| `render_mode="human"`     | 打开交互式 MuJoCo viewer。                                     |
| `randomize=True`          | 启用 `rand_full` 评估所使用的视觉随机化机制。  |
| `randomize_dynamics=True` | 在任务支持时启用动力学随机化。               |
| `seed=<int>`              | 设置任务环境的随机种子。                                          |

## OpenPI 客户端

`dexjoco_openpi_client` 软件包提供 DexJoCo 侧的 OpenPI 策略评估客户端。它会将 DexJoCo 的观测和动作适配到 OpenPI WebSocket 策略接口，包括摄像机映射、图像缩放、prompt 注入、状态切片，以及旋转向量/四元数动作转换。

在 `openpi` 环境中启动 OpenPI 策略服务器：

```bash
cd openpi
conda activate openpi
python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config water_plant \
  --policy.dir ../checkpoints/pi05_ckpts/water_plant/<exp_name>/<step>
```

在仓库根目录下的 `dexjoco` 环境中运行评估：

```bash
conda activate dexjoco
dexjoco-openpi-eval \
  --config=./configs/rand_obj/water_plant.yaml \
  --seed=0 \
  --port=8000
```

对于 `rand_full` 评估，请使用 `configs/rand_full/` 下的配置，并传入 `--rand-full`：

```bash
dexjoco-openpi-eval \
  --config=./configs/rand_full/bimanual_microwave_cook.yaml \
  --seed=0 \
  --port=8000 \
  --rand-full
```

评估视频和成功率标记文件默认会写入 `outputs/`。使用 `--output` 可以选择不同的输出目录。

评估配置位于 `configs/rand_obj/` 和 `configs/rand_full/` 下。每个配置都会定义 DexJoCo 任务、摄像机映射、语言提示词和机器人布局：

```yaml
env_name: water_plant
camera_mapping:
  base: front
  wrist: wrist
prompt: "Grasp the watering can and apply water to the plant."
robot_type: single_arm
```

对于双臂策略，摄像机映射使用 `base`、`wrist_left` 和 `wrist_right`，并将 `robot_type` 设置为 `dual_arm`。

`dexjoco-openpi-eval` 支持以下选项：

| 选项                            | 默认值        | 描述                                                                              |
| --------------------------------- | -------------- | ---------------------------------------------------------------------------------------- |
| `--config PATH`                   | 必填       | 位于 `configs/rand_obj/` 或 `configs/rand_full/` 下的评估 YAML。                       |
| `--seed INT`                      | `0`            | NumPy 和 Python random state 的随机种子。                                           |
| `--rand-full`                     | `False`        | 启用 `rand_full` 视觉随机化机制。                                     |
| `--randomize-dynamics`            | `False`        | 启用动力学随机化。                                                          |
| `--port INT`                      | `8000`         | OpenPI WebSocket 策略服务器端口。                                                     |
| `--host HOST`                     | `0.0.0.0`      | OpenPI WebSocket 客户端使用的主机地址。                                        |
| `--output PATH`                   | 自动生成 | 视频和成功率标记文件的输出目录。                               |
| `--render-mode {rgb_array,human}` | `rgb_array`    | DexJoCo 渲染模式。                                                                  |
| `--replan-ratio FLOAT`            | `0.8`          | 在请求新的动作块之前，要执行的 OpenPI 动作 horizon 的比例。 |
| `--episodes INT`                  | `50`           | 要运行的评估 episode 数量。                                                    |
| `--pad-state-dim46`               | `False`        | 为兼容性将策略状态表示填充到 46 维。                 |
| `--record-pressed-digits BOOL`    | 取决于任务 | 控制是否在 episode 输出名称中记录 iPad 数字输入。                 |

## 致谢

- 该仿真栈最初构建在 [Kevin Zakka](https://kzakka.com/) 的工作基础之上。
- DexJoCo 环境对这一基于 Gymnasium 的基础进行了适配和扩展。

## 许可证

该软件包中 DexJoCo 拥有的代码基于 MIT License 发布。捆绑在 `dexjoco/sim/envs/xmls` 下的第三方机器人和手部资源保留其各自的许可条款。
