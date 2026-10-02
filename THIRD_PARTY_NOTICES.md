# 第三方许可与来源说明

各第三方文件保留原许可证。本项目未来的代码或数据许可证不替代模型、网格、纹理和依赖的原有条款。

| 组件 | 许可证据与适用范围 |
|---|---|
| DexJoCo 上游自有代码 | MIT；原版权声明与完整许可保留在 `licenses/third_party/dexjoco/LICENSE`；项目改编代码需保留该声明 |
| DexJoCo 任务/场景素材 | 225 个待核验文件已与官方固定版本匹配；[准确来源与许可范围说明](licenses/third_party/dexjoco/README.md)；第三方模型条款仍独立核验，不统一指定 MIT |
| Franka Panda | Apache-2.0；`simulator/dexjoco/sim/envs/xmls/franka_emika_panda/LICENSE` |
| Wonik Allegro | BSD 条款；`simulator/dexjoco/sim/envs/xmls/wonik_allegro/LICENSE` |
| Sharpa 官方资产 | Apache-2.0；`simulator/teleoperation/sharpa_wave/sharpa-urdf-usd-xml/` 内的 `LICENSE.txt`、`NOTICE.txt` 与 `SOURCE_PROVENANCE.md`；不自动覆盖其他模型副本 |
| LinkerHand | Apache-2.0；`simulator/dexjoco/sim/envs/xmls/linkerhand_l20/SOURCE_LICENSE` 与 `SOURCE_PROVENANCE.md`；源几何来自官方 L25，内部历史键保留为 `linkerhand_l20` |
| LEAP 已匹配网格 | 官方 LEAP_Hand_Sim 的 MIT；[许可与 11 个随本版分发文件记录](licenses/third_party/leap/VERIFIED_FILES.json)，完整许可见同目录 `LICENSE.txt`；其余文件尚未核验完成 |
| Inspire 网格 | Dex-URDF 的 CC BY-NC-SA 4.0；[8 个原网格及 68 个改编网格的记录](licenses/third_party/inspire/README.md)；非商业分享须遵守署名和相同方式共享要求，剩余 URDF 链仍单独核验 |
| RoboSuite 已匹配场景资产 | RoboSuite v1.4.1 的 MIT 许可及其保留的第三方声明；[17 个随本版分发文件记录](licenses/third_party/robosuite/VERIFIED_FILES.json)，完整许可见同目录 `LICENSE`；不覆盖额外添加的纹理 |
| XHand | 官方 DexLatent 固定版本的 MIT；[62 个文件的来源记录](licenses/third_party/dexlatent/README.md)，其中 60 个网格被正式环境引用；生成文件的修改记录单独保留 |
| 项目文字纹理使用的字体 | DejaVu/Bitstream Vera 字体许可已保留在 `licenses/third_party/dejavu/NOTICE.txt`；不分发字体二进制，历史纹理来源见 `docs/PROJECT_ASSET_PROVENANCE.json` |
| 软件依赖 | MuJoCo、Gymnasium、NumPy、SciPy、PyTorch、Decord、Being-H 等由各自上游许可证约束；本项目不分发完整 Being-H 源码或基础模型权重 |
| π0.5 推理后端 | 使用者单独安装固定版本的 RLinf；本版只提供观测与动作的接口适配，不分发 RLinf 模型源码、训练补丁或权重。RLinf 的 Apache-2.0 条款见 `licenses/third_party/rlinf/LICENSE` |

`licenses/third_party/*/VERIFIED_FILES.json` 中列出的文件保留与上游一致的内容；文件名、路径或打包方式可能改变。其他转换、镜像和改编文件的来源链仍单独核验。

机器人/任务资产的详细状态见 [资产核验说明](docs/ASSET_LICENSE_AUDIT.md)。正式 42 个环境实际引用文件的逐项记录见 [运行时资产报告](docs/ASSET_RELEASE_REPORT.json)。未列明或标记为未确认的文件不能因本项目用于学术研究而被视为已获得再分发许可。

RGB 示范数据含模型和纹理的渲染结果。该数据集的许可证与相关素材的适用条款需要另行确认；本说明不对全部数据指定 MIT 或 CC 许可。

当前私有候选包包含七手六任务所需最小资产集合。历史核验中未被本版引用的 LEAP/RoboSuite 文件未包含在候选包中，其记录保留在相应 JSON 的 `historical_verified_files_not_distributed` 字段。许可范围未确认的 282 个文件保留逐文件来源记录，当前并未取得新的上游许可确认；适用范围见 [许可说明](LICENSE_SCOPE.md)，后续处理方案见 [资产发布方案](docs/ASSET_PUBLICATION_OPTIONS.md)。
