# 参与 ReadCue

ReadCue 当前在验证中文朗读规范化是否存在值得自训的缺口，尚无 ReadCue 模型权重。欢迎可复现的错误报告、真实使用案例、数据来源核对和小范围代码改进。

修改前请阅读 [项目计划](docs/PROJECT_PLAN.md)、[实验协议](docs/EXPERIMENT_PROTOCOL.md)和[数据与许可](docs/DATA_AND_LICENSES.md)。提议新训练、模型架构或任务范围时，先说明需求、现成对照和可验证的差异；付费运行需要明确预算授权。

## 本地检查

先按 [README](README.md#lightweight-development) 克隆并进入仓库。核心包支持 Python 3.10 及以上，建议使用 Python 3.12 和独立虚拟环境；没有必装的模型运行依赖：

```sh
python -m venv .venv
```

激活环境后运行；Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`，Linux 使用 `source .venv/bin/activate`：

```sh
python -m pip install "setuptools>=77" wheel
python -m pip install --no-deps --no-build-isolation -e .
python -m readcue verify
python -m unittest discover -s tests -v
python -m readcue --help
```

测试使用本地样例和标准库，不需要 GPU、模型权重或付费服务。CI 配置覆盖 Windows、Linux 的 Python 3.10/3.12，并检查 wheel 和源码分发包的内容与安装；具体通过状态以对应提交的 Actions 结果为准。模型基线复现另见[中文 pilot 复现说明](docs/REPRODUCING_ZH_PILOT.md)；核心开发无需安装该推理环境。

## 提交内容与证据

- 保持修改范围清楚，在 PR 中写明问题、改动和验证结果。代码测试通过不代表模型更好。
- 分开报告论文成绩、数据审计和本项目实测。当前 72 条原创 pilot 未经人工逐条复核，不能称作真实用户样本或盲测。
- 推理只接收公开输入，不使用 gold 参考、类别或理由路由。保留失败输出，不为提高分数静默修改参考。
- 历史 pilot 的冻结文件、脚本哈希与报告用于追溯。新实现和修正保留版本及原因，不覆盖旧证据或把重新调过的样例称作未见测试。
- 新数据说明来源、授权和处理过程。不要提交模型、缓存、第三方原始语料、密钥或私人用户内容；代码的 Apache-2.0 不替代数据与模型的条款。

GitHub Actions 由 Dependabot 按月提出更新。既有实验的依赖锁文件应随新的复现实验有意更新，并保留原版本记录。

源码打包与未来 Hugging Face 权重发布的要求见[发布说明](docs/RELEASING.md)；二者是不同交付。文档入口见[文档导航](docs/README.md)。
