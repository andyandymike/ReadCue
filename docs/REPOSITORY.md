# GitHub 仓库维护

仓库：[andyandymike/ReadCue](https://github.com/andyandymike/ReadCue)。2026-09-28 的管理设置通过 GitHub REST API 修改并重新读取核对，见[设置记录](maintenance/github-settings-2026-09-28.json)。

## 定位与协作

仓库保持 public、默认分支 main，简介与主题说明中文文本规范化、小模型和可复现研究；尚未发布 ReadCue 模型权重。Issues 配有错误报告和研究建议模板，添加 research、evaluation、data 标签。文档统一放在版本管理内的 docs/，关闭目前空置的 Wiki 与仓库 Projects。

PR 使用 squash 合并，默认采用 PR 标题与正文，合并后自动删除源分支。main 的基础规则防止删除和强推；正常提交仍可进行，适合当前个人维护方式。它不是“所有提交先通过检查”的门禁；需要严格 PR 工作流时再增加必需检查。

## 检查与依赖

CI 在 Windows/Linux 的 Python 3.12 上检查核心包与历史证据，不下载模型，不安装 CUDA 或完整实验锁文件。并发运行自动取消被新提交替代的旧任务。工作流令牌默认只读，不能批准 PR；仓库要求 Actions 固定完整 commit SHA。

Dependabot 按月检查 Actions 版本。依赖漏洞提醒和安全更新已启用，密钥扫描与推送保护保持开启。历史实验锁文件和冻结脚本的变更需要版本化复现实验，不能为解决依赖提醒而静默改写历史记录。

CI 使用 GitHub 标准托管 runner；本项目不配置付费 GPU 作业。模型试验与未来训练仍按项目预算和数据协议单独管理。

## API 与发布边界

管理设置使用 `PATCH /repos/{owner}/{repo}`、topics、actions/permissions、vulnerability-alerts、automated-security-fixes、labels、rulesets 等 REST 接口。参考：[仓库 API](https://docs.github.com/en/rest/repos/repos)、[Actions 权限 API](https://docs.github.com/en/rest/actions/permissions)、[分支规则 API](https://docs.github.com/en/rest/repos/rules)。

源代码、原创评测、报告和维护记录可以进入 Git；原始第三方数据、权重、虚拟环境、缓存和运行目录由忽略规则隔离。上传前验证 Git 索引文件清单与大小，远端提交后核对完整 tree SHA。Hugging Face 权重发布不属于仓库初始化，不自动触发。
