# 数据来源与许可

本文件是来源记录与待办，不把数据许可证自动解释成训练权重许可证。仓库Apache-2.0覆盖ReadCue原创代码与项目文档，不覆盖第三方材料、数据、基座或将来权重。

## 固定研究来源

当前已转入中文可行性验证，以下 UGTPhon 表格保留为历史候选研究来源，不代表中文训练数据已选定。

| 来源 | 已核实的声明 | 用途与边界 |
| --- | --- | --- |
| [UGTPhon](https://github.com/naver-ai/UGTPHON) | 各语言单独许可 | 官方commit `da2b3ee84e15e98ba48b388f7ee291ba4c3579b2` |
| [越南语子集](https://github.com/naver-ai/UGTPHON/blob/da2b3ee84e15e98ba48b388f7ee291ba4c3579b2/vi/LICENSE) / [ViLexNorm](https://github.com/ngxtnhi/ViLexNorm) | CC BY-NC-SA 4.0 | 首轮研究候选；非商业、署名、相同方式共享等条款需遵守 |
| [英语子集](https://github.com/naver-ai/UGTPHON/blob/da2b3ee84e15e98ba48b388f7ee291ba4c3579b2/en/LICENSE) | 研究用途、禁止商用；原始来源没有明确许可 | 已作数据审计，不作为许可已核清的默认发布路线 |
| [韩语子集](https://github.com/naver-ai/UGTPHON/blob/da2b3ee84e15e98ba48b388f7ee291ba4c3579b2/ko/LICENSE) | 声明Apache-2.0 | 非首轮范围；辅助发音资源仍各自有来源条款 |
| [IPA-Dict](https://github.com/open-dict-data/ipa-dict#credits) | 仓库MIT，但明确第三方数据保留原条款 | 越南语源自vPhon与Ho Ngoc Duc词表；原词表授权的一手核验仍待完成 |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | Apache-2.0 | 候选基座，训练前固定revision；不覆盖微调数据 |

这些是2026-09-28调研记录，不代表所有第三方许可问题已完成。vPhon软件许可不能自动推导其输出或训练权重适用同一许可。

## 仓库保存与获取

原始语料、模型权重、缓存和训练输出不提交Git；`.gitignore`已有对应排除规则。归档保存我们写的审计脚本、聚合统计、哈希清单和研究说明。公开来源的短词例子保留来源说明，不将第三方材料重新授权为Apache-2.0。

审计脚本可按固定版本重新获取en/vi六份JSON，约38.3 MB，详见[复查归档](../research/ugtphon-2026-09-28/README.md)。数据获取与使用仍受其原始条款约束。

## 权重发布前需完成

- 列明实际使用的基座、数据、词典及其固定版本和来源。
- 对辅助词表尚未核明的来源给出明确处理决定。
- 确定模型权重的具体发布条款，并单独说明限制，不直接沿用仓库代码许可。
- 提供UGTPhon及上游数据引用，标注独立复现/改进身份。
- 记录是否重新发布任何词典或数据派生文件，以及适用的归因和共享要求。

项目目标是发布研究权重；目前不存在已获许可确认的商业模型，也不存在已经上传的HF模型。

## 中文本地 pilot（2026-09-28）

- [项目原创72条情境样例](../eval/zh_reading_pilot_v1/README.md)：assistant 编写，未复制第三方语料，尚无人类逐条审核；专门声明按 Apache-2.0 发布。这是可提交的原创评测材料，不属于排除的原始第三方语料。
- [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)：代码仓库 Apache-2.0；本轮安装 PyPI 1.2.0 作规则对照。
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)：模型卡 Apache-2.0，仅用于本轮本地推理。
- [leeoxiang/qwen3-0.6b-zh-tn](https://huggingface.co/leeoxiang/qwen3-0.6b-zh-tn)：模型卡 CC-BY-NC-4.0，注明训练含标贝非商业数据。本轮作非商业研究对照，不将该权重重新授权或上传至ReadCue。
- [CPP](https://github.com/kakaobrain/g2pm)、[CVTE-poly](https://github.com/NewZsh/polyphone)、[FlatTN](https://github.com/thuhcsi/FlatTN)仅调研为候选，未纳入当前pilot、未用于训练；各自数据来源和独立条款仍待核清。不能由代码许可证推断数据授权。

两份基线权重保存在忽略的 models/ 中，按具体 HF revision 下载并核对大文件上游 SHA-256。清单记录版本、文件大小与哈希，不将模型卡条款迁移到原创测试句或未来模型。
