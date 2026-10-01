# 中文公开文本试采

2026-09-28 小规模试采已完成。MDN 中文文档可以稳定获取，并保留固定版本、原文和出处。本轮验证的是**收集能力和数据链路**，不是训练效果，也没有收集到真实用户的朗读纠错反馈。

## 实测结果

| 来源 | 获取结果 | 提取结果 |
| --- | --- | --- |
| MDN 中文 | 60 篇；原文共 625,918 字节 | v2 保留 1,570 条去重候选句；182 条含阿拉伯数字，996 条含拉丁字母 |
| 中文维基百科 | 0 篇；2 次请求失败 | 官方 API 和一次普通页面请求均在 TLS 证书校验阶段报证书过期，未收到 HTTP 响应或正文 |

MDN 批次使用 7 次 GitHub API 请求取得目录及许可信息，61 次原文请求取得 60 篇文档与许可文件，均成功。前期路径探测另行保留，不计入批次请求数。维基百科没有关闭证书校验或改用代理，其他计划页面未尝试。

原始材料和提取结果保存在仓库外：`E:\CodexRuntime\ReadCue\data\zh-source-probe-2026-09-28`。未上传数据、未调用收费服务、未训练模型，也未改变现有冻结的 72 条 pilot 或其结果。

## 来源与选择方法

- 上游：[mdn/translated-content](https://github.com/mdn/translated-content/tree/259254e64c3552a0cdf7bac7bf541e7390f004e6)。固定中文仓库 commit：`259254e64c3552a0cdf7bac7bf541e7390f004e6`。文档中的英文翻译参考 commit 不作为中文正文版本。
- 从 HTTP 指南、JavaScript 指南、HTTP 状态码三个目录，各按文件路径的 SHA-256 排序取前 20 篇；选择在读取正文前完成。这是有意选择的技术领域样本，并非互联网中文的随机样本。
- v2 分组：HTTP 指南 529 条，JavaScript 指南 1,002 条，状态码 39 条。JavaScript 约占 64%，代码表达式和英文标识较多。
- 有数字只是粗筛条件，不代表该句存在朗读歧义或模型错误。当前也没有得到消费电子型号、价格、地址等日常场景的充分覆盖。

## 提取版本和证据

`mdn/batch-v1` 保存首次抓取的 60 篇原文、目录选择、请求记录、原文与 Git blob 哈希、采集器源码及 1,740 条候选句。其中 192 条含数字。复核发现句末括号归属、缩进列表续段及 Markdown 标记处理问题。

`mdn/batch-v2` 从同一批经过哈希验证的本地原文离线提取，新增网络请求为 0。v1 完整保留。v2 保护行内代码和转义符的字面含义，吸收连续句末闭合括号及引号，排除所有缩进正文和无法可靠消除的强调标记。与 v1 相比，1,552 条文本完全相同；188 个旧文本 ID 不再出现，18 个新文本 ID 出现，净减少 170 条。删除与改写的差异均可由两版记录追溯，不涉及读法标签修改。

主要文件：

- `raw/`：原始字节、原始许可文件。
- `documents.jsonl`、`selection.json`、`requests.jsonl`、`metadata/`：固定来源、标题、署名、选择记录、网络请求和目录响应。
- `sentences.jsonl`：可读文本、原始段落、源行号、完整上下文、显示文本偏移、转换说明、许可与未标注状态。
- `duplicates.jsonl`：完全相同文本的重复出处；没有静默丢弃出处。
- `collector-source.py`、`summary.json`：确切提取程序及其哈希、运行状态和统计。
- `inspection-sample.jsonl`：按 ID 排序取最多 50 条含数字、50 条不含数字的检查样本，非代表性抽样。
- `inspection-preview.md`：本地另外挑选的 20 条阅读预览，有意展示数字、时间和协议表达，不作为质量通过率的分母。

句子来自公开出版的文档。移除了链接和部分格式标记，折叠了空白，并按中文句末标点切分；保留原始段落和行号以供核对。宏、图片、代码块、列表、表格和复杂 HTML 不作为普通正文。这个保守提取器不是完整 Markdown 渲染器，可能漏收有效文本；语境完整性仍需逐条筛选。

所有记录均为 `human_reviewed: false`、`annotation_status: unlabeled`。程序核验和 assistant 抽查不等于人工逐条审核。没有生成标准读法、用户偏好或错误标签。源文档公开已久，无法证明任何对照模型没有见过它们，因此不称为无污染盲测。

已全量核对 60 份原文的字节数、SHA-256 和 Git blob SHA-1、1,570 条记录的原始行切片与显示文本偏移，以及 68 份批次请求记录的本地响应文件。`verification.json` 保存核验摘要。按 CI 的 `PYTHONPATH=src` 设置运行 28 个本地测试通过，两个既有文件符号链接子例因 Windows 权限跳过。现有冻结评测材料与历史结果未改变。

## 许可

MDN 的[固定版本许可声明](https://github.com/mdn/translated-content/blob/259254e64c3552a0cdf7bac7bf541e7390f004e6/LICENSE.md)与[官方署名说明](https://developer.mozilla.org/en-US/docs/MDN/Writing_guidelines/Attrib_copyright_license)将文档正文声明为 CC BY-SA 2.5。记录保留 Mozilla Contributors、页面标题、固定来源和逐行贡献历史链接，并说明提取变换。代码示例另有条款；本轮排除了独立代码块，候选句仍可能包含简短行内标识和表达式。

第三方原文和派生句子不适用 ReadCue 的 Apache-2.0 代码许可。如后续发布数据，需要随附其来源、署名、修改说明和适用许可；本次抓取成功不等于未来训练权重的许可已获确认。尚未授权或执行 Hugging Face 发布。

## 复现

Python 3.10+；联网采集还需可用的 GitHub CLI 和 GitHub API 访问。输出必须是仓库外尚不存在的目录，采集器拒绝写入仓库内部。

```powershell
python tools/collect_zh_mdn.py --revision 259254e64c3552a0cdf7bac7bf541e7390f004e6 --per-group 20 --output E:\CodexRuntime\ReadCue\data\mdn-new-capture
```

当前程序直接产生修正后的提取结果。若需要复现历史 v1，使用该批次保存的 `collector-source.py`。对已有完整抓取离线重新提取：

```powershell
python tools/collect_zh_mdn.py --revision 259254e64c3552a0cdf7bac7bf541e7390f004e6 --offline-from E:\CodexRuntime\ReadCue\data\zh-source-probe-2026-09-28\mdn\batch-v1 --output E:\CodexRuntime\ReadCue\data\mdn-reextract-new
```

每篇下载上限 1 MiB，三个固定目录每组最多 30 篇，请求间有间隔。发生错误即停止并保留状态和已获取的证据，不覆盖旧批次。此命令不会生成读法标签或运行模型。

下一步可从含数字的候选中先筛选语境完整、确实有朗读难点的文本，并按源文档分组划分用途，避免同篇文档的近邻句跨训练与评测集合。当前尚未创建任何训练或评测划分。
