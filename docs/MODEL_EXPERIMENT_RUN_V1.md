# 首轮 ReadCue 模型实验记录

日期：2026-10-01。对应[启动协议](MODEL_EXPERIMENT_V1.md)与[固定读法政策](MODEL_TASK_V1.md)。本次记录与此前规则、第三方模型和音频探索分开保存。当前运行产物在维护者本机 `E:/CodexRuntime/ReadCue/runs/model-experiment-v1-2026-10-01/`，这些路径不是公开下载地址。

## 数据与隔离

原有160条项目原创候选，经成功绑定审计、逐条政策复核与分组，形成118条训练、29条验证、13条待审。4条原审计未定参考继续拒绝，另9条存在政策或来源边界歧义；均未进入监督目标。17条的参考集合有明确版本化修改记录，其中13条改变训练目标、4条收窄格式备选，原文与旧参考字节保持不变。38个整体模板组按固定种子分配，ID、原文、归一文本、数字模板和组别的跨角色交集为零。训练与验证均来自已看材料，不能称为独立测试。

独立标注任务根据固定政策另写60条原创评测，并只查看训练/验证的输入与组别做重复检查。首次正式冻结前剔除5条模板近似案例，保留初稿和剔除记录；最终55条：普通TN19、语境读法20、原文保留16，44个局部目标、4条预登记备选参考。所有参考为助手标注，未经用户逐条确认；这些是合成情境，不代表自然评论分布或随机盲测。实现者在训练与选点完成前只核验评测文件哈希，训练入口只读取训练/验证及不含参考正文的冻结证明。

| 固定材料 | SHA-256 |
| --- | --- |
| 任务提示配置 | `09eaef3f2d88416abf7f2b1ce00b4eba5b0e849f05f2ae8783e3be6e711bb904` |
| 训练/验证 manifest | `906b83491d88f768974b2cacd85705c0b5ae78c60a27e19d21c36e26cc71186b` |
| 新评测证明 | `378b6ab1535b5fc9aaf8edfa8efcfd2d2124c3385a4698803bcf36aa1295b3a9` |
| 新评测输入 | `9b2e35a03a283a64cbf98b9d643f7dcf6ec5d605433cd2f5032a2504e5dee69f` |
| 新评测参考 | `6317fb6ac5ca085cc466841233318122d912e2e5386f482a0c8116afcd7d31c3` |

B站评论未进入监督目标。新原创派生数据在独立数据包中登记Apache-2.0范围，不改变第三方材料条款。基座沿用本地Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`，其[固定版本许可](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE)为Apache-2.0；所有资产与历史冻结清单核对。

## 环境与运行方式

使用WSL Python3.12.3、PyTorch2.6.0、Transformers4.57.6、PEFT0.17.1、Accelerate1.10.1，GPU为RTX2060 6GB。独立训练环境复用原推理环境的已固定依赖，只在新环境安装训练包；原环境包版本未变。新增wheel由配置的镜像获取，与官方PyPI逐文件SHA-256核对后离线安装。详见[环境锁文件](../requirements-model-v1.lock)。这些训练依赖不会成为核心工具的强制依赖。

以上描述首轮训练当时的环境。2026-10-01 后续维护已将训练依赖合并到唯一 GPU 环境，并将旧训练路径改为兼容入口；新运行增加实际源码身份、依赖锁与缓存目录检查。原训练、选点、评测及其源码快照保持原样，下面的成绩不由维护后代码重新计算或替换。当前布局与启动约束见[存储说明](STORAGE.md)。

实现遵循固定LoRA配置：r8、alpha16、dropout0.05，仅q_proj/v_proj；基座FP16，训练参数FP32，micro-batch1、累积8，AdamW1e-4、warmup0.03后恒定学习率、梯度裁剪1。只监督目标及EOS，提示和输入标签被屏蔽。训练与验证按目标token加权；模板前缀、完整正文与目标不截断。上限仍为3个epoch、100次更新、2小时及5GiB，按验证损失选点，平手较早。

预检已完成一次参数更新和保存/重新加载，重载前后目标token损失同为1.2019225358963013；耗时160.74秒，峰值显存预留1,707,081,728字节。成功预检的临时adapter在记录哈希后移除，正式训练重新初始化；该损失仅证明保存/重载一致，不是模型效果评测。

可复现入口：

```sh
# 使用自己的合法、版本化数据包及基座目录；各输出必须为新目录。
python tools/prepare_model_data_v1.py --audit-dir /path/to/reference-audit --policy-config configs/model_task_v1.json --review-recipe /path/to/review-recipe.json --output /path/to/dataset
python scripts/run_model_experiment.py preflight --project /path/to/ReadCue --model-dir /path/to/qwen3-0.6b --task-config /path/to/ReadCue/configs/model_task_v1.json --dataset /path/to/dataset --output /path/to/experiment/preflight --budget-root /path/to/experiment
python scripts/run_model_experiment.py train --project /path/to/ReadCue --model-dir /path/to/qwen3-0.6b --task-config /path/to/ReadCue/configs/model_task_v1.json --dataset /path/to/dataset --heldout-proof /path/to/heldout/freeze-proof.json --output /path/to/experiment/train --budget-root /path/to/experiment
python scripts/run_model_experiment.py predict --project /path/to/ReadCue --model-dir /path/to/qwen3-0.6b --task-config /path/to/ReadCue/configs/model_task_v1.json --inputs /path/to/heldout/inputs.jsonl --adapter /path/to/experiment/train --output /path/to/experiment/adapted --budget-root /path/to/experiment
```

基座对照使用同一predict命令但省略`--adapter`。预测只接收id/text，完整模型输出保留，无答案提取或规则回退。训练成功并完成选点后才能运行`tools/score_model_experiment.py`；它核对训练时的冻结证明、各运行输入、配置与所选adapter身份，避免预检或错配预测冒充正式比较。自动跨度对齐遇到不确定时保留待核查，不把字符差异直接当成语义错误；正文语义另做隐藏方法名的输出审查。

```sh
python tools/score_model_experiment.py --project /path/to/ReadCue --training-run /path/to/experiment/train --heldout /path/to/heldout --heldout-proof /path/to/heldout/freeze-proof.json --base /path/to/experiment/base/predictions.jsonl --adapted /path/to/experiment/adapted/predictions.jsonl --rules /path/to/experiment/rules/predictions.jsonl --output /path/to/experiment/evaluation
```

## 实际训练结果

正式训练成功结束：118条训练、29条验证，3个epoch、45次优化器更新，1,146,880个可训练LoRA参数，耗时557.68秒。峰值CUDA实际分配1,663,348,736字节，峰值预留3,927,965,696字节；没有触发显存、数值、时间或存储上限。

| epoch | 验证目标token平均交叉熵 | 最终选择 |
| --- | --- | --- |
| 1 | 0.42834741535712045 | 否 |
| 2 | 0.46380793724926606 | 否 |
| 3 | 0.3937384251868819 | 是 |

选择第三轮第45次更新的adapter，重新加载原始固定基座及该adapter后，验证损失仍为0.3937384251868819。正式训练保存第一轮与第三轮两份检查点；选定产物由本机`train/selected_adapter.json`定位，权重为`train/checkpoints/step-0045-epoch-3/adapter_model.safetensors`。损失用于预定选点，不当作独立评测分数。

## 固定评测结果

选点后一次运行55条新合成情境，四组均55条正常完成，无缺失、异常、空输出或截断。基座与微调模型使用相同输入、提示、非思考模板和greedy设置，不做答案提取、规则后处理或失败回退。基座推理耗时200.42秒，微调模型245.74秒；包含各自加载开销，不作为公平吞吐基准。

| 方法 | 严格／表面整句匹配 | 普通TN | 语境读法 | 原文保留 | 保留组改动 |
| --- | --- | --- | --- | --- | --- |
| 返回原文 | 16/55（29.1%） | 0/19 | 0/20 | 16/16 | 0/16 |
| 未微调Qwen3-0.6B | 0/55（0%） | 0/19 | 0/20 | 0/16 | 16/16 |
| ReadCue LoRA v1 | 28/55（50.9%） | 7/19 | 7/20 | 14/16 | 2/16 |
| 固定规则audit-corrected-v1 | 26/55（47.3%） | 0/19 | 10/20 | 16/16 | 0/16 |

这批严格与表面匹配计数相同。与同条件基座比较，微调新增28条整句匹配、没有已匹配条目退步；基座大量输出英文翻译、解释或未授权改写，0/55反映它在这套固定任务提示下没有完成输出契约，不代表其所有提示、解码或其他任务的能力。固定规则不具备通用数字展开阶段，因此整体分数不能用来证明模型在每种能力上优于规则：语境组模型为7/20，低于规则的10/20。

规则与微调模型逐条配对：共同匹配17条、共同未匹配18条，微调新增匹配11条（7条普通TN、4条语境），退步9条（7条语境、2条保留），净增2条。两种方法各有不同失败项，不能把净增2条表述为全面优于规则。

字符对齐初检中，44个局部目标的匹配数为原文0、基座0、微调18、规则11。基座有13个目标无法可靠对齐，按未完成留在分母。这是机械匹配，不等同于语义或发音正确。

### 隐藏方法名的逐例审查

独立助手只读取固定政策及匿名候选，逐条检查55×4＝220个输出，完成后才由主任务用映射表汇总方法身份。它没有读取模型选择、其他预测目录或原始分数；审阅记录和方法说明保留。这是助手文本复核，不是用户逐条批准或听音测量。

| 方法 | 审阅确认的局部目标完成 | 新增语义错误句 | 保留组语义错误句 | 目标操作以外改动句 |
| --- | --- | --- | --- | --- |
| 返回原文 | 0/44 | 0/55 | 0/16 | 0/55 |
| 未微调基座 | 4/44 | 48/55 | 14/16 | 55/55 |
| ReadCue LoRA v1 | 19/44 | 7/55 | 1/16 | 3/55 |
| 固定规则 | 11/44 | 0/55 | 0/16 | 0/55 |

语义错误包括事实或限定条件丢失、数量表达受损、编号/版本/实体字符串被改变，不把阿拉伯数字未展开、逐位读法或所有格式差异自动视为语义错误。模型的7条分别为HEV1-002、012、016、020、022、026、049；最后一条属于未知实体字符串被改写，七、八本身的数值没有变。这一计数包含不同严重程度，不能把7条视为等量风险。

局部完成不保证整句正确：基座的4个局部目标主要是署名/表情删除，但其余正文仍存在改写。微调的19个目标比机械匹配多1个，是HEV1-030确实省略了表情，却多加句号；仍保留原始整句未匹配，不回补分数。HEV1-054把含混狐狸表情展开为“狐狸”，属于本轮保留政策违规；未擅自补出褒贬，不据此另算语义错误。基座HEV1-009涉及“暂借外馆”的借用方向，审阅保留不确定，未强计为确定语义错误，仍留在55条分母。所有参考与首轮成绩不变。

微调模型的27条未匹配中，包括普通数量逐位读、缩写未分字母、百分数表达不符合政策、漏掉应展开的符号，以及实际数量错误。例如3.25公里被改成“三又二分之五公里”，35元变成“三百五元”；211工程被误写为“二十一世纪工程”。保留组的两条改动分别是未知型号`QX7/P8`被部分展开、含义未定的狐狸表情被写成“狐狸”。这些错误说明当前权重尚不适合直接替代保守输出。

## 产物与边界

本机`evaluation/summary.json`保存原始自动评分，`*.score.json`保存逐组与逐例记录；`semantic-review.jsonl`保存匿名助手判断，`final-summary.json`在不改原评分的前提下关联审阅结果与规则配对分析。预测、完整原始输出和token均留在各方法目录。所选权重SHA-256为`60c50ba5cbaa9e4cb5952eafa7d4e3d7284c131151bb477ea65db088837bb984`，4,602,248字节，是需搭配固定基座的LoRA adapter，不是独立完整模型。使用说明与能力边界见[本地模型卡](MODEL_CARD_V1.md)。

本轮提供“有限监督可以改变这批任务行为”的初步证据，同时暴露数字、缩写覆盖和保留边界不足。55条均为助手原创合成材料，参考未经用户逐条确认；不是自然评论分布、随机盲测或统计稳定的泛化证明。本轮没有测量发音准确率或TTS提升，也不将历史试听确认移用到新权重。评测后未更改政策、提示、参考、规则、超参数或重训；后续实验应使用新版本数据和另行冻结的评测。

所有输出和权重保持本地，未上传模型或产生外部费用。工程验证与语义审查分开记录，不以测试通过代替效果判断。

## 工程验证

Windows与WSL Linux各运行275项测试通过；Windows跳过2项需要符号链接权限的检查，这两项在Linux实际执行通过。训练与评分相关新增测试包括数据资格、提示标签屏蔽、累积与选点、保存/加载身份、失败分母、机械跨度和冻结证据绑定。7份原pilot冻结文件哈希未变，正式运行的源码哈希与训练前冻结一致；实际identity运行也与评分所用原文逐条一致。

独立结果复核通过44项身份、哈希及快照检查；四套评分在内存重新计算后与保存结果逐字段一致。完整记录为本机`result-integrity-review.json`，这项检查没有参与匿名语义判断。

发行检查使用`tools/check_distribution.py`检查wheel/sdist内容，再在两个新环境中离线安装并验证CLI、冻结材料、真实Unicode输入、源码字节一致和无需ML依赖的核心导入。打包不包含数据、权重、运行输出或环境；本次实际构建和安装收据单独保存于本机运行目录的`verification/`。这些检查不等于已向PyPI或HF发布。
