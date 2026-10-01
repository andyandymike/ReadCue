# 离线评论数据准备与审阅导出

第一版接入已取得的 `yunyunfanfan/job_change_as_ai` 原始 CSV，使用 Python 标准库，沿用现有 CLI。它负责可复现的数据整理；不抓取网站、不调用模型、不生成读法答案、不训练。首批结果见[运行报告](../reports/bilibili-data-v1/REPORT.md)。

## 运行准备流程

按项目 README 安装核心包后，从 checkout 运行。也可在 PowerShell 临时设置 `$env:PYTHONPATH = 'src'`，直接使用源码。

```powershell
python -m readcue data prepare --source-root 'E:\CodexRuntime\ReadCue\data\newer-bilibili-probe-2026-09-29\github\yunyunfanfan__job_change_as_ai\files' --config configs/bilibili_recent_v1.json --output 'E:\CodexRuntime\ReadCue\data\bilibili-review-new'
```

这里的 E 盘路径只是当前机器的捕获位置。其他机器把 `--source-root` 替换为自己的原始文件目录；配置内文件路径相对于这个目录。源 CSV、README 和采集脚本必须与[固定来源配置](../configs/bilibili_recent_v1.json)的字节数、SHA-256、Git blob SHA 全部一致。命令只读本地文件，没有下载入口。

`--output` 必须为新目录，位于 Git checkout 外，或确实被 Git 忽略。相同配置重跑时换一个输出目录。失败留下 `run.json` 和已形成的证据，不覆盖原始文件或旧运行。

## 每一步做什么

1. **校验与导入。** 全部来源文件先验哈希，再从同一份已验字节解析 CSV。使用严格 UTF-8 与 CSV 语法，保留文本中的换行、空白、全角字符、短词、重复字、表情、大小写和 @ 提及。作者 UID、昵称、等级不进入整理记录；原文中自带的提及和标识不自动改写。
2. **质量审查。** 标识符、时间、列数错误等业务坏行进入 `rejected.jsonl`，保留原行和位置。不可恢复的 CSV 结构或 UTF-8 错误使该次运行失败。相同评论 ID、相同内容的重复记录折叠到一个 canonical record，同时保留全部来源；同 ID 内容冲突时全部隔离。不同 ID 的相同文本保留并标记。
3. **启发式抽样。** 时间窗为北京时间 `[2024-01-01, 2026-01-01)`。数字、拉丁字母、重复字符、方括号表达及 Unicode 符号只是审阅线索；没有匹配这些规则的记录进入普通对照池。空白、替代字符或异常控制字符记录不参与抽样，排除条件写入 `selection.json`。URL、长文本和短文本保留标记。
4. **视频覆盖与关联。** 固定 seed，使用 SHA-256 排序并按视频轮流取样；候选目标 112 条、对照 28 条。某池不足时如实记录，不从另一池补满。同视频及跨视频完全相同的文本形成连通分组，供未来划分使用。本轮所有材料是探索材料，没有自动制造训练／测试划分。
5. **待审阅产物。** 输出 Markdown 清单和 JSONL 标注副本。所有状态初始为 `pending`，参考答案为空。此时不生成 `inputs.jsonl` 或 `cases.jsonl`。

使用这些特征会改变抽样分布，不能用样本中的错误比例推算全站错误率。含数字、字母或表情也不代表应当规范化，更不代表现有模型会读错。

## 本来源的上下文限制

已核查固定版本上游采集脚本：它把每条二级回复的 `parent_rpid` 固定设为根评论 ID。1,439 条回复的真实直接回复对象因此没有保留下来。配置以 `context.parent_semantics=root_only` 明示这一点；整理记录、审阅材料和未来导出来源信息都会保留该限制。

清单中的上下文只能称为“已保留的根评论”，不能当成完整多轮对话。需要真正上文才能确定读法的记录应标为 `uncertain`，不能借根评论猜一个答案后按纯文本任务评分。

## 产物和审阅方法

优先使用本地审阅页，避免手工改 JSONL。以下是当前机器这批140条的入口；其他机器替换为自己的 prepare 输出目录：

```powershell
$env:PYTHONPATH = 'src'
python -m readcue data review --prepared 'E:\CodexRuntime\ReadCue\data\bilibili-pipeline-2026-09-29\prepare-02'
```

在浏览器打开命令打印的本机地址，保留终端运行；按 Ctrl+C 停止。默认端口8765，冲突时加 `--port 0` 自动选空闲端口。页面使用本机字体，无外部脚本或模型请求；关闭服务后，已保存标注仍在文件中。重新启动会读取同一文件并定位首个待审条目。完整用法见[本地审阅页](REVIEW_WORKBENCH.md)。

| 文件 | 用途 |
| --- | --- |
| `records.jsonl` | 完整合法评论、原文、来源、时间、质量标记与关联组 |
| `rejected.jsonl` / `duplicates.jsonl` | 异常与重复处理证据；不是静默删行 |
| `groups.jsonl` | 同视频与完全重复文本连通组；当前没有数据集划分 |
| `review.md` | 可直接阅读的候选、根评论上下文和出处 |
| `review.jsonl` | 固定抽样清单，不编辑 |
| `annotations.jsonl` | 可填写的副本，仅修改每行 `review` 对象 |
| `selection.json` / `summary.json` | 规则、seed、实际选择和聚合统计 |
| `config.json` / `sources.json` / `code/` | 配置快照、来源校验回执和实现源码快照 |
| `manifest.json` / `run.json` | 固定文件哈希、阶段状态、失败原因；标注副本不列为不可变文件 |

审阅页填写的是 `annotations.jsonl` 的 `review` 对象；也可使用文本编辑器填写：

- `status`：`approved`、`excluded` 或 `uncertain`。后两者必须记录原因。
- `reviewer`：实际审核者标识；软件只记录声明，不能证明某个人实际完成了审核。
- `acceptable_outputs`：批准时填写非空的合理输出列表，允许多个读法。
- `track`：批准时选择 `core_tn`、`context_reading`、`preservation`。这里的 context_reading 指本条文本内的语境，不代表增加楼中楼作为模型输入。
- `family`：批准时填写表达家族，供离线分组。
- `context_required`：批准导出纯文本案例时必须明确为 `false`；若依赖丢失的上文则保留为不确定材料。
- `notes`：审核说明；排除和不确定记录必须填写。

`preservation` 的参考需精确保留原文。原文、ID、来源、抽样标记和上下文字段不能在审核时改动；需要纠正来源时创建新的版本，保留旧版。

## 完成审核后导出

```powershell
python -m readcue data export --prepared 'E:\CodexRuntime\ReadCue\data\bilibili-review-new' --review 'E:\CodexRuntime\ReadCue\data\bilibili-review-new\annotations.jsonl' --output 'E:\CodexRuntime\ReadCue\data\bilibili-reviewed-new'
```

导出检查准备包哈希、审核 ID 全量一致和原文字段不变。仍有 `pending`、缺少参考、未知上下文依赖等情况会拒绝形成评测输入，并保存失败状态。

成功导出只将 approved 条目写入：

- `inputs.jsonl`：严格只有 `id/text`，接现有 `readcue run`。
- `cases.jsonl`：另外保存参考、track 和 family，接现有 `readcue score`。
- `provenance.jsonl`：来源、组、上下文限制和审核者，不传入模型。
- `review_decisions.jsonl`、`submitted_annotations.jsonl`：全部审核决定及提交的原始字节，包含 excluded / uncertain，避免把难例悄悄移出记录。

导出包记录原始抽样、提交审核、准备包和各产物哈希。它是新探索材料，不能使用仅接受历史72条案例的旧 `archive` 命令冒充 v1。后续根据这些材料调参或训练时，需另建独立评估。TTS 接入和听感审查仍是后续工作，文本匹配不等于发音收益。

## 扩展范围

当前只有一种明确的 CSV 来源适配，不设置爬取调度、插件注册或自动训练。增加来源时先核日期、许可、文本变换及字段含义，再添加相应适配与配置；破损 ID 不自动“修复”为看似精确的 ID。共享哈希与输入校验沿用现有核心，历史 runner、评分器、冻结样例和成绩保持不变。
