# 代码与实验边界

ReadCue 目前是小规模研究项目，已增加一次有界LoRA实验所需的独立执行入口，没有通用训练平台或产品服务。可维护性首先意味着实验身份、数据来源和失败记录可靠；只为当前任务补充明确边界。

## 两条代码路径

**历史复现路径固定不动。** `eval/zh_reading_pilot_v1/freeze.json` 记录三份脚本、协议和样例的哈希。`scripts/run_zh_baseline.py`、`scripts/score_zh_pilot.py`、`scripts/prepare_zh_models.py` 保留原始字节。已有预测、元数据和评分不因代码整理而覆盖。不要修改旧哈希来让一次重构通过校验。

**新增工程入口可以演进。** `src/readcue` 提供零强制运行依赖的工具包，入口是 `python -m readcue` 或 `readcue`。命令接受 `--project` 指定项目 checkout，默认当前目录。安装的包不夹带模型、数据或历史报告；评分和复现仍需该 checkout。

| 文件 | 职责 |
| --- | --- |
| `src/readcue/artifacts.py` | 有界内存哈希、上游 Git blob / LFS 校验、校验后发布下载文件、原子 JSON 记录、模型身份及冻结文件核验 |
| `src/readcue/inference_contract.py` | 独立于模型和评分器的 id/text 校验、保留字符串值的 JSONL 序列化 |
| `src/readcue/pilot_adapter.py` | 历史 pilot 的系统枚举与冻结神经运行配置校验 |
| `src/readcue/evaluation.py` | 校验后加载 v1 原始评分器；校验每个系统身份、revision和结果哈希，再归档 |
| `src/readcue/baselines.py` | 严格 id/text 输入、模型资产预检、子进程调用历史 runner、持久状态和错误日志 |
| `src/readcue/cli.py` | 薄命令入口，组织 verify / fetch / run / score / archive |
| `src/readcue/comment_sources.py` | 固定本地CSV与证据文件核验、原文保真导入和异常记录 |
| `src/readcue/data_pipeline.py` | 离线质量标记、确定性抽样、审阅快照与无参考泄漏的导出 |
| `src/readcue/review_contract.py` | 不依赖 HTTP 的人工标注、建议绑定与轮次状态契约，供审阅服务和导出共用 |
| `src/readcue/model_experiment.py`、`scripts/run_model_experiment.py` | 新模型实验的预检、训练、独立adapter加载与原文推理，单独保存资源上限和运行身份 |
| `src/readcue/model_scoring.py`、`tools/score_model_experiment.py` | 选点后的离线对照评分、局部目标与配对变化；核验训练、评测证明及预测身份 |
| `scripts/archive_zh_pilot.py` | 未被旧冻结清单绑定的兼容入口，转到新归档实现 |

历史整句评分继续核对并调用原评分器，历史神经推理仍使用已审查的runner。新模型实验另有明确的训练和推理实现，复用冻结加载兼容检查、模型资产验证及独立输入契约。Torch、Transformers和PEFT仅在明确执行模型命令时导入，核心安装仍无强制ML运行依赖；旧pilot代码、哈希和成绩不因新实验改变。

## 新模型实验入口

首个本地adapter已完成训练并独立重载，权重与运行产物仍在仓库外，尚未公开上传。模型实现和实际运行证据的区别、依赖环境及复现命令见[首轮运行记录](MODEL_EXPERIMENT_RUN_V1.md)。

`scripts/run_model_experiment.py`提供三个子命令：`preflight`只用训练集的一条样本验证前后向、保存及重载，成功后移除临时adapter并保留哈希；`train`从固定基座和种子重新初始化，只读登记的训练／验证文件，按目标token平均验证损失选点；`predict`独立加载基座及可选adapter，只接收id/text，保留完整生成内容与失败状态。训练只读评测冻结证明中的哈希，不打开评测正文或参考。

`tools/score_model_experiment.py`是另一个离线入口，在训练完成和选点后读取评测参考与已保存预测。它核对训练绑定的冻结证明、同一任务配置、各系统输入以及被选adapter，再计算对照结果；不调用模型或更改参考。局部文本对齐及整句分数不代替语义或试听审核。

执行身份与目录管理属于这个入口的边界：实际加载的包、辅助模块和入口文件必须与 `--project` 对应源码一致；脚本不能来自另一份 checkout。运行前保存实际路径、源码快照和锁文件，并在导入 ML 库前核对 `requirements-model-v1.lock` 的全部精确版本；未登记的额外包不在这项核验范围内。

三个阶段都必须提供同一实验的 `--budget-root`，输出是其下独立的新目录。共享缓存固定在 `runtime-cache-v1/cache/`，临时文件在各阶段 `tmp/`，均纳入 5 GiB 定期检查；不接管无归属的已有缓存，拒绝链接和无法读取的预算内容。每个阶段须用新进程启动；推荐脚本入口，模块入口使用 `python -B -m readcue.model_experiment`，防止 Python 在配置路径前产生预算外字节码。失败仍保留独立收据，不覆盖旧结果。本机环境、目录布局与保留策略见[存储说明](STORAGE.md)。

## 本地使用

核心安装不下载 Torch、CUDA、模型或规则包：

```sh
python -m pip install --no-deps --no-build-isolation -e .
python -m readcue verify
python -m readcue score --cases eval/zh_reading_pilot_v1/cases.jsonl --predictions reports/zh-pilot-v1/tn.jsonl --output runs/inspection/tn.score.json
```

上述安装需要已有 setuptools>=77 和 wheel。历史 GPU / WeText 环境仍按 [复现说明](REPRODUCING_ZH_PILOT.md)的独立锁文件准备；`requirements-zh-pilot.lock` 是那次 Linux 环境记录，不是所有使用者都要安装的核心依赖。

新运行使用一个全新目录，即使导入、模型加载或预热失败，也会留下 `run.json` 和已产生的日志。不会把未运行的模型当作能力零分：

```sh
python -m readcue run --inputs eval/zh_reading_pilot_v1/inputs.jsonl --baseline identity --output runs/new-pilot/identity
python -m readcue fetch --baseline tn --revision 97f7860d117f651f9aeba1bc6558eec7c378c958 --destination models/zh-tn
python -m readcue run --inputs eval/zh_reading_pilot_v1/inputs.jsonl --baseline tn --model-dir models/zh-tn --revision 97f7860d117f651f9aeba1bc6558eec7c378c958 --output runs/new-pilot/tn
```

`fetch` 会访问网络并下载资产；`run` 不自动安装依赖或下载模型。示例只是使用说明，不表示已重新运行神经模型。运行前核对模型 repo、revision、每个文件内容；历史缓存可与冻结清单中的逐文件 SHA-256 对照，新下载则对每个文件使用 HF 返回的 LFS SHA-256 或 Git blob SHA-1 来源证明。Git blob 哈希不是普通文件 SHA-1，包含长度头。

模型目录只允许清单登记的普通文件和 `readcue_manifest.json`。额外的 `chat_template.jinja`、adapter、子目录、符号链接或 Windows 重解析文件都被拒绝，避免加载器读到未核验的覆盖配置。下载也拒绝这类残留内容；工具不会替用户删除它们，应使用独立的干净资产目录。

公开运行入口保存提交的输入字节，并在必要时生成供历史执行器读取的兼容副本：Unicode 行分隔字符在 JSON 字符串中转义，解码后的 id/text 不变。运行记录分别绑定提交输入和实际执行输入，仍拒绝参考或分类字段；原历史执行器和冻结哈希保持不变。

四个系统都运行完后，归档接受 `runs/new-pilot/<baseline>/predictions.jsonl` 这种新目录，也接受旧版同目录四组命名文件：

```sh
python -m readcue archive --run runs/new-pilot --output reports/new-pilot
```

该归档命令只接受原始 pilot 的输入哈希和四个已记录的基线身份，包括 WeTextProcessing 1.2.0 与协议中三个明确为 false 的规则选项；新样例可单独调用 `score`，不能冒充原始 pilot。原始预测可以包含逐例失败，但未完成的外层运行不能伪装为完整归档。归档写入失败会留下明确失败状态，不覆盖旧报告。

神经结果还须满足冻结的 CUDA/FP16、生成参数、随机种子、提示与解码方式。合法的 CPU 推理可以单独运行和评分，但不作为同一配置归档。包装运行的外层记录和输入证据随新归档保留，避免清理运行目录后丢失来源上下文。

## 扩展方式

本轮模型实验已建立独立案例、协议和runner，没有修改历史v1。后续实验仍先固定案例与协议，给新来源、配置、提示和评分建立独立清单；当前输入/预测JSONL和逐句失败字段足以支持这个范围，不要求建立通用数据框架。

目前没有抽象模型注册中心、插件发现、训练调度、自动发布或自动 GPU 申请。是否训练、花费多少以及数据能否发布仍按项目计划决定。

真实评论的第一条轻量数据准备流程已接入 `readcue data prepare/export`，见[操作说明](DATA_PIPELINE.md)。它复用原始哈希和输入契约，单独保留来源、审核与上下文限制；只有已填写审核的批准记录才导出 `id/text` 输入。本轮材料是探索集，不更改v1或自动划分训练／测试。JSONL读取按LF分隔，导出时转义Unicode行分隔符而保留原字符串值，以兼容冻结的历史runner。

## 验证

`readcue data review` 为上述离线流程提供本机浏览器审阅入口。`review_server.py` 使用标准库HTTP服务，`review.html` 随核心包分发；随机路径与同源校验限制访问，写入只接受单条 `review` 对象。冻结材料、原文和来源字段每次保存都校验；可编辑标注使用版本检查、原子替换和旧版本快照。页面不连接模型、不生成参考，不改变既有导出和推理隔离边界。详见[审阅页说明](REVIEW_WORKBENCH.md)。

CI 不安装模型依赖，也不联网下载或运行神经模型。新增测试用已有四组共 288 条预测逐项重算，要求所有历史评分和汇总不变；另覆盖系统/revision错标、gold字段和重复id拒绝、子进程失败持久记录、伪网络的 Git/LFS 校验、校验失败不发布文件，以及损坏缓存不覆盖。identity 子进程检查只返回原文。

```sh
python -m unittest discover -s tests -v
```

已有七项评分/解析测试继续保留。新包的任何整理都不应通过“更新参考、覆盖旧预测、改写冻结哈希”获得通过。

## 近期探索代码的处置（2026-10-01）

当前CLI的`run`仍调用历史基线，`data`只提供用户要求的准备、审阅和导出。`comment_policy*`、`reading_policy`、`context_reading_policy*`、`semantic_emoji_policy`及对应组合脚本属于探索实验，不是自训ReadCue推理接口；没有默认接入审阅页或`readcue run`，不能把它们的词表增量当成模型能力。

旧v4发现宽学校语境误判和跨阶段保护缺项，撤回其作为后续可靠处理入口的地位，保留原文件用于历史复现。审计修正只撤销不成立的推断、补上保护和记录独立版本，不继续扩词。具体修正、可用入口与验证见[工作复查](WORK_REVIEW_2026-10-01.md)。

新的模型研究遵循[模型实验v1](MODEL_EXPERIMENT_V1.md)：模型直接接收原文，仅用训练目标监督，不以规则预处理结果替换评测输入或自动充当gold。规则对照与模型输出分别保存。原审阅UI、用户标注及其版本恢复功能保留。

2026-10-01 代码检查阶段的维护收拢了已经存在的契约：输入格式、人工标注、轮次绑定和pilot配置。探索入口的源码清单补入共享依赖，策略与模型计算不变；重现旧成绩仍使用对应外部冻结包，不能将维护后的源码哈希补写回历史包。此后的独立模型实验复用这些小模块，新增执行源码与历史探索分别记录。
