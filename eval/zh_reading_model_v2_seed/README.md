# ReadCue v2 首批原创开发样例

日期：2026-10-27。用途：核对训练覆盖和划分方案，作为后续正式v2材料的候选。

`numeric.draft.jsonl` 和 `context.draft.jsonl` 分别保存数值与语境的助手原创草稿，每份36条。每项技能3个模板组、每组2条；建议角色合计48条训练、24条验证。草稿经审查如需修改，另存版本并记录，不覆盖初稿。

每条包含原文、候选整句目标、合法备选、技能、组别、建议用途和理由。`provenance` 明确助手作者、非用户逐条审核、原创材料许可与开发用途。`training_ready=false`、`independent_test_eligible=false`：本目录不是可直接传入训练入口的数据包，也不承担新独立评测。

政策沿用 [MODEL_TASK_V1.md](../../docs/MODEL_TASK_V1.md)，原始标点和空白保留。完整覆盖、合并与新评测方案见 [MODEL_DATA_V2.md](../../docs/MODEL_DATA_V2.md)。

离线检查无需ML依赖：

```powershell
.\.venv\Scripts\python.exe -B tools/check_model_v2_seed.py
```

仅检查本目录不能证明跨版本排重完成。带历史材料的检查显式指定 `--v1-dataset` 与 `--v1-heldout-inputs`；后者严格只接受id/text，不读取旧评测参考。检查输出是结构和重复风险报告，不确认标签语义或训练资格。

## 原创材料声明

本目录的原创样例及目标由ReadCue项目助手为本次任务编写，没有采集第三方原文，也没有用现有模型预测作标签。原创样例按仓库根目录 [Apache-2.0 LICENSE](../../LICENSE)提供，与第三方数据或将来模型权重的条款分开。本声明仅覆盖本目录原创内容，不覆盖任何外部对照材料。样例为定向合成文本，不代表自然评论分布；助手审阅不等同于用户批准或听音验收。
## 本批文件与实际状态

- [SAMPLES.md](SAMPLES.md)：72条原文与候选目标的阅读版。
- [numeric.draft.jsonl](numeric.draft.jsonl)、[context.draft.jsonl](context.draft.jsonl)：原始草稿，未改写。
- [assistant_review.json](assistant_review.json)：另一助手对全部72条的逐例语义审阅、原文／目标哈希与覆盖局限；全部accept，仅指候选目标符合当前政策。
- [check.draft.json](check.draft.json)：第一版机械检查收据。
- [check.reviewed.json](check.reviewed.json)：检查器补入“每个技能的训练／验证都必须含转换正例”后的新收据；含检查器源码哈希。
- [v1-coverage.json](v1-coverage.json)：原数据的角色／类别／家族计数与来源哈希，不含旧原文或目标。

两次机械检查均为0阻塞、0告警；实际核对72条新样例、147条v1训练／验证输入及55条旧评测输入。两次记录均保留，未覆盖。机械收据的 `semantic_review_complete=false` 只描述检查器不判断语义；语义审阅另见上列文件，二者不能互相替代。

本批候选为48训练／24验证，41转换／31保留。合并原v1后的预计数量166／53不等于正式数据包已冻结。新评测尚未编写；本批不得转作独立测试。
