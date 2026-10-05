# 公开访问验收（2026-10-05）

项目负责人于 2026-10-05 授权公开评测代码与数据集，公开入口如下：

- [评测代码](https://github.com/darenrenjian/DexJoCo-X-public)
- [数据集](https://huggingface.co/datasets/darenrenjian/DexJoCo-X)
- [项目网站](https://darenrenjian.github.io/DexJoCo-X-website/)
- [论文](https://arxiv.org/abs/2610.03278)

未登录请求两个仓库均返回 HTTP 200，代码通过 HTTPS 匿名克隆。数据下载测试移除令牌环境变量、使用空 Hugging Face 配置目录并禁用隐式令牌，下载敲钉子任务 TAR，自动解压并完成 350 条示范、1,754 个文件的 SHA-256 校验。冻结下载 revision 不变。

从匿名克隆源码构建 wheel 并安装到独立目录，使用此前验证的 Python 3.11 依赖环境。安装后的 801 个资产均与冻结校验值一致；从安装目录导入仿真模块，完成 Allegro 敲钉子环境初始化、256×256 RGB 观察和有限数值的物理步进。完整 42 环境的原验收记录见 [随包验收](BUNDLED_ASSET_VALIDATION.md)。

网站 Code 和 Dataset 入口已启用，线上桌面、手机、字体、引用卡片、图片放大和视频检查通过。完整工作仓库、策略与编码器权重、手套映射及旧数据历史归档仍为私有。

本轮只更新公开状态、说明和入口，不修改原始示范、模型几何、第三方许可核验结论或冻结下载版本。详细记录见 [JSON](PUBLIC_ACCESS_VALIDATION.json)。
