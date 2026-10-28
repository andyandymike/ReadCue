# ReadCue 文档导航

当前已完成第二轮本地ReadCue LoRA训练、独立重载和新45条固定合成情境对照：v2整句匹配32/45、原v1为28/45、规则25/45；v1→v2新增4条匹配，无匹配退步。训练使用166条、验证53条，共3轮、63次更新，权重未公开上传。分项局限、完整性核验及独立匿名审查见[v2运行记录](MODEL_EXPERIMENT_RUN_V2.md)。首轮旧55条的28/55等结果继续保存在[首轮记录](MODEL_EXPERIMENT_RUN_V1.md)及[v1模型卡](MODEL_CARD_V1.md)，不与本批分母混用。

| 要做什么 | 从哪里开始 |
| --- | --- |
| 安装核心工具，运行无 GPU 示例 | [项目 README](../README.md#lightweight-development) |
| 了解当前目标与继续条件 | [项目计划](PROJECT_PLAN.md)、[首轮模型实验](MODEL_EXPERIMENT_V1.md) |
| 查看v2数据准备和正式划分 | [v2覆盖矩阵与方案](MODEL_DATA_V2.md)、[首批原创开发样例](../eval/zh_reading_model_v2_seed/README.md)；正式数据为166训练／53验证 |
| 复查v2模型实验 | [v2固定协议](MODEL_EXPERIMENT_V2.md)、[实际运行与评估](MODEL_EXPERIMENT_RUN_V2.md)、[v2模型卡](MODEL_CARD_V2.md)、[聚合收据](model-v2-results.json)；保留启动前阻塞历史与冻结收据 |
| 查看固定v2的训练／验证诊断与下一版范围 | [诊断报告、技能表和独立不匹配审查](MODEL_DEVELOPMENT_DIAGNOSTIC_V2.md)、[聚合收据](model-v2-development-diagnostic.json)；不是独立泛化测量 |
| 使用本地v2研究发布候选 | [候选构建、独立加载与GPU执行等价检查](MODEL_RELEASE_CANDIDATE_V2.md)；未公开上传，条款与发布位置尚未最终确定 |
| 运行首轮模型预检、训练、推理和离线评分 | [实际运行与复现入口](MODEL_EXPERIMENT_RUN_V1.md)、[固定标注政策](MODEL_TASK_V1.md)；模型依赖单独安装，核心工具仍无强制ML依赖 |
| 查看最近工作的撤回、修正与缺项 | [近期复查与处置](WORK_REVIEW_2026-10-01.md) |
| 使用当前审计规则或检查已有TN候选 | [规则入口](../scripts/run_audited_rules.py)、[修正及验证边界](WORK_REVIEW_2026-10-01.md)；不自动加载模型 |
| 查看已完成的对照与局限 | [中文 pilot 报告](../reports/zh-pilot-v1/REPORT.md) |
| 查看138条已审核评论的四组结果 | [评论对照报告](../reports/bilibili-data-v1/BASELINE_REPORT.md)、[独立冻结协议](BILIBILI_EVAL_PROTOCOL_V1.md) |
| 查看评论规则与专用TN的组合效果 | [组合结果与未解决问题](../reports/bilibili-data-v1/PIPELINE_REPORT.md)、[组合冻结协议](COMMENT_PIPELINE_PROTOCOL_V1.md) |
| 检查英文、混合版本读法并试听 | [读法结果与4对本地音频](../reports/bilibili-data-v1/READING_REPORT.md)、[v2冻结协议](READING_PIPELINE_PROTOCOL_V2.md) |
| 查看学校编号、u1s1修复与30条定向真实评论 | [语境修订结果与音频索引](../reports/bilibili-data-v1/CONTEXT_REPORT.md)、[v3协议](CONTEXT_READING_PROTOCOL_V3.md)、[v3.1修订协议](CONTEXT_READING_PROTOCOL_V3_1.md) |
| 查看有词义表情保护与学校表述扩展 | [v4语义结果、待审项与局限](../reports/bilibili-data-v1/SEMANTIC_REPORT.md) |
| 核查数据、参考与评分口径 | [样例说明](../eval/zh_reading_pilot_v1/README.md)、[实验协议](EXPERIMENT_PROTOCOL.md) |
| 查找更新的 B 站评论与弹幕材料 | [日期核验、实际获取与来源限制](NEWER_BILIBILI_SOURCES.md) |
| 整理评论、抽样和导出已审阅案例 | [数据 pipeline 操作说明](DATA_PIPELINE.md)、[首批运行结果](../reports/bilibili-data-v1/REPORT.md) |
| 逐条审阅、保存进度和恢复标注 | [本地审阅页](REVIEW_WORKBENCH.md) |
| 核对助手预填的读法与分类 | [AI 预标注口径与人工核对](AI_PREANNOTATION.md) |
| 复现四组本地基线 | [复现流程](REPRODUCING_ZH_PILOT.md) |
| 修改代码或检查来源条款 | [参与说明](../CONTRIBUTING.md)、[架构](ARCHITECTURE.md)、[数据与许可](DATA_AND_LICENSES.md) |
| 查看模型、依赖、缓存与临时文件位置 | [当前存储布局与清理规则](STORAGE.md)、[本轮优化记录](maintenance/2026-10-01-runtime-optimization.md) |
| 准备源码或未来模型发布 | [发布说明](RELEASING.md)、[本次开源检查](maintenance/2026-10-01-open-source-review.md) |

以下为历史探索结果，后续行动以当前计划和复查处置为准。

公开仓库提供代码、原创pilot和报告；第三方原文、逐条审核记录、模型及语音仍在仓库外。报告里的`E:/CodexRuntime/...`链接只能在维护者本机使用，不是公开下载地址或安装所需目录；复现时使用自己的合法来源材料和本地路径。历史音频确认也只对应报告绑定的片段，不代表整体语音效果已验收。

2026-10-01后续结果：v3修复旧第45、100条，开发集格式归一匹配为115/138（严格64）；v3.1仅修订一处“本硕”边界，首批新增24条中的有参考项从11/22变为12/22，另2条待审，这批已转为开发材料。补充6条定向真实评论新旧版本均为3/6，保护拒绝计失败。两批同源材料分开报告，108份新增语音已生成；用户随后确认展示的学校编号与u1s1两段处理后音频读法正确，其余音频待核对。未训练或付费。

随后v4完成268条：旧236条仅5条文本或状态改变，其余231条相同，未出现旧匹配退步。开发集达到116/138（严格65），已看补6达到5/6；新24条原创控制为9/20参考、4条待审，新8条真实评论仍为2/6参考、2条待审。词义未定和参考冲突单列，不改标签补分。用户随后确认展示的v4“九八五硕博”与“大便一样的文档”两段音频读法正确，独立记录并保留为回归对照；其余新音频仍待核对，前次已接受两片段保留。

[UGTPhon 研究归档](../research/ugtphon-2026-09-28/README.md)与 [archive/](archive/) 保存早期越南语候选路线，不能替代当前中文计划。[仓库配置记录](REPOSITORY.md)和[存储维护记录](STORAGE.md)供维护者追溯；其中本机盘符、迁移与恢复路径不属于安装要求。
