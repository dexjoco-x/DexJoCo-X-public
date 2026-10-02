# Versioning

- `paper-v1`：论文 Table I 所用的训练数据、checkpoint 和评测结果。其训练发生在 XHand 微波炉左手安装修复之前。
- `dataset-v1.0-corrected`：公开数据候选，包含修复后的 50 条 XHand 微波炉示范。
- `runtime-v9`：与修复安装一致的评估和仿真环境。

`paper-v1` checkpoint 保持私有；历史记录声明它没有在 `dataset-v1.0-corrected` 上重新训练。若需要严格的数据–模型一致性，应使用修复数据重新训练 Native、FAAS 和 DexLatent，并重新执行完整 42×150 评测。

## 兼容标识

`linkerhand_l20` 是数据清单、环境文件和 checkpoint 合约中已经冻结的内部键。审计发现，该目录中的左右手源 URDF 实际由官方 LinkerHand L25 文件修改而来，并非官方 L20 文件。为保持已有实验可复现，当前版本不重命名内部键，也不替换几何模型；论文和用户文档统一写作 `LinkerHand`。准确型号经项目负责人确认后，应在下一主版本中迁移内部键，并提供旧键到新键的兼容映射。
