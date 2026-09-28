# 中文可行性对照协议 v1

日期：2026-09-28。仅运行现成模型，无 ReadCue 训练结果。[原越南语协议](archive/VI_EXPERIMENT_PROTOCOL_2026-09-28.md)供历史复查。

## 任务与参考

本轮评估完整句子朗读规范化文本，不评估发音或音频。读法与来源见[样例说明](../eval/zh_reading_pilot_v1/README.md)。72 条 assistant 原创样例分为 core_tn、context_reading、preservation，每类24条、6个情境家族。

参考是有限合法读法集合，尚未人工复核。未匹配不必然读错。外来词、缩写和商品型号按约定保留书面形式，文本保留成功不证明下游 TTS 会读对。

## 冻结和隔离

- cases.jsonl 包含离线参考和分类，首次运行前固定哈希。
- inputs.jsonl 仅 id/text，推理入口拒绝其他字段，不按参考、类别、rationale 路由。
- 公共提示在首次运行前固定。通用 Qwen 接收显式任务提示，专用 TN 使用作者接口，规则系统使用固定配置；这是现成系统的实用比较，不是同训练条件的因果消融。
- 不根据输出修改提示、规则、参考后覆盖首轮成绩。修正必须版本化并保留旧结果。
- UGTPhon test 前期已查看；中文pilot也是公开探索材料，均不能称为未见盲测。

## 对照与推理

| 对照 | 固定配置 |
| --- | --- |
| identity | 返回原文 |
| WeTextProcessing | PyPI 1.2.0，remove_erhua=False、remove_interjections=False、traditional_to_simple=False，其他默认 |
| 专用 TN | leeoxiang/qwen3-0.6b-zh-tn，固定 revision，作者 span-edit 接口，greedy |
| 通用 Qwen | Qwen/Qwen3-0.6B，固定 revision，统一 system prompt，关闭 thinking |

通用 Qwen 使用模型卡的 non-thinking 建议参数 temperature=0.7、top_p=0.8、top_k=20，每例固定随机种子。仅一轮采样，不声称统计稳定或代表最优提示。专用模型用其卡片 greedy。输出上限均160 token，无结束token的输出算截断。

模型 FP16、batch=1、本机 CUDA，禁止远程自定义代码，仅加载 safetensors；规则系统在CPU运行。先用固定非pilot句子预热，分开记录加载开销及逐句耗时。本地延迟不是同计算量的公平硬件比较。

专用模型按作者顺序锚点解析；空编辑表示原样保留。非法锚点、空替换或格式失败保留原输出、标记失败；回退原文恰好正确也不算模型成功。

## 评分与判断

1. 严格整句 EM：等于任一参考。
2. 表面格式归一化 EM：NFC、全角ASCII、中英等价标点、空白连续段折成一个空格、移除汉字相邻空格。保留拉丁单词边界、字母大小写、汉字、数字、下划线和百分号等内容；同时保留严格分数。
3. preservation 的原文改动、失败数量、合计违规率分别报告。
4. 各 track 独立报告，补充 family 全部通过数；关联情境不视为独立随机抽样。
5. 原始输出、解析结果、状态、耗时全部保留；缺失、空输出、截断和异常都计失败，不能移出分母。

不将 EM 当音素错误率、句子听感或真实用户收益。人工复核须区分参考未覆盖、读法偏好、工具契约与实际内容错误，不仅凭合并分数宣布优劣。

## 复现与保存

scripts/prepare_zh_models.py 下载固定版本公开模型；scripts/run_zh_baseline.py 执行无gold输入推理；scripts/score_zh_pilot.py 离线评分。

第三方原始语料、模型权重和缓存不提交Git。原创样例按专属许可发布，本地免费推理配置、聚合和逐例证据可归档。无付费授权。
