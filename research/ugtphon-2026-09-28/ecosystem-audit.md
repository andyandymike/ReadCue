# UGTPhon 成品、接入及发布审计

检查日期：2026-09-28。只读公开来源，没有安装、下载模型、运行 TTS、训练或付费调用。

## 官方实物与现有模型

GitHub API 固定 UGTPhon main 为 `da2b3ee84e15e98ba48b388f7ee291ba4c3579b2`。文件树包括三语 train/dev/test JSON、语言许可证、README、图和 moderation 元数据；没有训练代码或权重。HF API exact search `ugtphon` 当前返回空；不能由此证明不存在别名权重。小模型权重的公开复现具有研究复用价值，但下载量或广泛需求没有可靠证据。

- https://github.com/naver-ai/UGTPHON
- https://huggingface.co/api/models?search=ugtphon&limit=30
- https://github.com/lingjzhu/CharsiuG2P
- https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100

CharsiuG2P 已有多语言 ByT5 权重、训练代码和词级 G2P 支持。UGTPhon 成品差异应是非规范文本的上下文处理、避免误改及明确的未见形式评测；不能宣传为第一个可下载的多语言 G2P 模型。

## 接入现实系统

### Kokoro/Misaki

Kokoro `KPipeline.generate_from_tokens` 允许直接输入 phoneme string；`KModel.forward` 使用模型 vocab 映射字符，未命中的字符会被滤去。因此接口存在不等于任意 IPA 直接兼容。Misaki EN_PHONES 明确其英语使用49个符号，包括以 A/I/W/Y 表示双元音、以 ʤ/ʧ 合并辅音等约定，与通用 IPA 不完全相同。必须做音素转换与未知符号断言。当前读取的标准 KPipeline language map 没有越南语；Misaki 自身 README 有越南语资源链接，不能将两者混成 Kokoro 官方模型支持越南语。

- https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py
- https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py
- https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md
- https://github.com/hexgrad/misaki

### Piper

当前 `PiperVoice.phonemize` 的 eSpeak 路径支持 `[[...]]` raw phonemes；另有 `phonemes_to_ids` 和 `phoneme_ids_to_audio`。仍要符合所选 voice 的 phoneme_id_map、声调、停顿和训练约定。可作为研究接入路线，但本轮没有读入声音模型或生成音频，不能声称实际变好。

- https://github.com/OHF-Voice/piper1-gpl/blob/main/src/piper/voice.py

### 越南语现有前端

VieNeu-TTS 当前公开 v3 Turbo 明确使用 SEA-G2P。SEA-G2P 是可独立使用的 Rust 规则/词典前端，包含越南语数字、日期、单位和英语混合读法。它是现实部署比较对象，不是只与裸基座大模型比较即可。其 README 速度是作者自报，不作为本审计实测。UGTPhon 没有证明在该流水线上改善语音或延迟。

- https://github.com/pnnbao97/VieNeu-TTS
- https://github.com/pnnbao97/sea-g2p

## 发布许可边界

- 越南语 UGTPhon 与 ViLexNorm：明确 CC BY-NC-SA 4.0。研究用途路线最清楚；不能把整个项目无条件标成 Apache-2.0 商用。
- 英语 LICENSE 明言源 LexNorm2015 无明确许可证、依据引用惯例提供，且限制研究、禁止商用。不能将其说成一个标准 CC 许可。
- 韩语数据文件声明 Apache-2.0，但 IPA-Dict 的 Korean pronunciation 来源是 Wiktionary CC BY-SA；不同组件需分别记录。
- IPA-Dict 仓库 MIT 说明明确保留第三方来源原条款。越南语来自 vPhon 和 Ho Ngoc Duc 词表；上轮一手词表页未成功获取，本轮未补齐该原始来源授权。软件 GPL 本身不等于其输出或训练权重自动 GPL。数据条款与训练权重适用条款也不能自动等同，应在发布前形成具体 provenance 说明。
- Qwen2.5-0.5B 与 ByT5-small 基座已有 Apache-2.0 标注，但基座许可不覆盖训练数据。

来源：
https://github.com/naver-ai/UGTPHON/blob/main/vi/LICENSE
https://github.com/naver-ai/UGTPHON/blob/main/en/LICENSE
https://github.com/naver-ai/UGTPHON/blob/main/ko/LICENSE
https://github.com/ngxtnhi/ViLexNorm
https://github.com/open-dict-data/ipa-dict#credits
https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/LICENSE

## 成本解释

论文每个方法 run 约24 V100 GPU小时，未拆前置 IPA 训练等阶段；全部实验约400 V100 GPU小时，API另计。当前 Google Cloud Iowa 页面 V100 16GB 为 $2.48/GPU小时，配套VM另计。

- 作者一次 run 资源量：$59.52。
- 四次相当规模训练：$238.08；这是算式，不保证能覆盖两模型对照、前置阶段与重跑。
- 论文400 GPU小时参考：$992，仅GPU，不是本人项目报价。
- 建议初轮只比较 train-only 字典、一个普通微调基线、一个组合方法，预算 $200–500 为规划估计；完整多种子/多语复现、人工检查不包含在内。

https://cloud.google.com/products/compute/gpus-pricing?hl=en

## 我的发布定位判断

可作为个人可负担、可独立复用的研究 checkpoint 项目。没有必要为发布而改新架构。最有说服力的模型贡献将是：不依赖 gold is_nc、处理词典未见形式、避免错误改写标准词，同时保留标准读音。若只是复现论文，也应诚实标注 unofficial reproduction，明确语言、输入跨度协议、训练词典和原始/清理后评测结果，不宣称是已证明提升听感的通用 TTS 前端。
