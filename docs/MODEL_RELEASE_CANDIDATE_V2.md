# ReadCue v2 本地研究发布候选

日期：2026-10-28。状态：本地候选，未上传、未发布；最终 adapter 发布条款尚未确定。创建候选不等于授权上传，也不将仓库代码或基座的 Apache-2.0 自动用于 adapter。

## 当前候选与保留记录

当前候选为 `E:/CodexRuntime/ReadCue/releases/readcue-v2-candidate-2026-10-28-r2/`。构建器只创建全新目录；所有原始运行、检查点、模型、数据和评分保持不变。

初版 `E:/CodexRuntime/ReadCue/releases/readcue-v2-candidate-2026-10-28/` 保留为 **superseded / non-release**。初版便携脚本对结果路径进行词法检查，`checks/../adapter/new` 或链接别名可能绕过写入边界。修正版先拒绝链接／reparse point，再解析实际路径并检查输出位置；两项专项回归已加入测试。没有覆盖或删除初版，不应使用其推理入口。

候选包含 23 个有哈希的 payload 文件：原选中 adapter 目录的全部四个文件、冻结任务与标注政策、训练依赖锁及历史 pilot 锁、基座文件清单与许可原文、项目代码许可、真实 v2 模型卡、训练／来源摘要、冻结加载和推理源码、便携入口及一条无参考演示输入。完整 payload 清单为候选根目录的 `bundle-manifest.json`；`checks/` 专门保存另外生成的本地验证记录，不属于 payload。

不复制基座权重或 tokenizer；使用者须另外取得相同 revision 的完整本地基座目录。不包含训练／验证原文、gold、原始数据集、逐条评测预测、密钥或运行缓存。候选中的训练来源摘要仅记录角色计数、配置、选择结果和文件哈希，不包含监督目标。原 PEFT README 也按原字节保留，其中的本机路径和占位字段不是正式发布模型卡；外层 README 和 `docs/MODEL_CARD_V2.md` 说明实际状态。

## 固定身份与加载契约

- 基座：`Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`。便携入口根据冻结清单核验完整基座文件的大小和 SHA-256，不根据目录名判断身份。
- 基座 LICENSE SHA-256：`832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`。
- 原选点：`model-experiment-v2-2026-10-27/train/checkpoints/step-0063-epoch-3/`，根据原 `selected_adapter.json` 核验 metadata。
- `adapter_model.safetensors`：4,602,248 字节，SHA-256 `27ab01f82e6811cd48ebb846cdf31c1d9adad746aa0370ec6ae8dd77f50b0f75`。
- 任务、政策和模型依赖锁均从训练快照复制，并与原运行、adapter metadata 及当前同版本文件交叉核对。

构建器从训练代码快照逐字提取 `load_local_tokenizer_and_config`、`prompt_tokens`、`_predict`、依赖核验及 id/text 输入合同。保留新版 tokenizer 字段兼容映射、完整词表 ID、added-token special 标志、FIM 单 token 和默认 RoPE 参数检查，不重写 tokenizer 或 config 文件。

正式预测仍为原任务 system prompt、`enable_thinking=False`、greedy、最多 1024 新 token、CUDA FP16、SDPA、原 EOS／padding 规则和原始解码。没有答案提取、额外清理、规则回退或输入截断。输入只允许 `id` 和 `text`；ID 不进入模型提示。空输出、异常和截断保留在结果记录中。

CPU `check` 明确屏蔽 CUDA，只核验、加载，不生成文本。加 `--load-model` 会将同一基座及 adapter 绑定到 CPU；这证明可读与可加载，不证明 CUDA 输出等价、文本正确或音频效果。模型锁是已记录的 Linux Python 3.12 环境，包含 CUDA 和继承实验依赖，不能据此承诺任意 CPU 平台均可安装。

## 构建与核验

在保留原运行的条件下，以另一个全新输出目录复现构建：

```sh
python tools/build_model_research_bundle.py \
  --project /path/to/ReadCue \
  --run-root /path/to/model-experiment-v2-2026-10-27 \
  --base-dir /path/to/fixed-qwen3-0.6b \
  --output /path/to/new-candidate
python tools/build_model_research_bundle.py \
  --verify /path/to/new-candidate --base-dir /path/to/fixed-qwen3-0.6b
```

同样的来源字节产生同样的 payload 和清单；构建清单没有写入当前时间或输出绝对路径。清单是完整性检查，不是发布者的数字签名。命令不安装依赖、不联网、不训练、不上传。

已在现有 WSL 环境运行 18 项专项测试，18 通过、0 跳过，包括真实 symlink、路径逃逸、拒绝覆盖、文件篡改、意外 payload、固定函数原文和无 gold 输入。

`-r2` 已完成 CPU 全模型加载：23 个 payload 及固定基座文件核验通过；112 个 adapter tensors、1,146,880 个参数全部有限并成功绑定，base 为 FP16、adapter 为 FP32、可训练参数为 0。完整词表 ID、special 标志、FIM 和 RoPE 核验通过，锁中全部依赖版本一致。没有执行生成。收据为候选中的 `checks/cpu-load-20261028/receipt.json`；加载阶段耗时 74.82 秒，不含先行完整文件哈希检查。

`-r2` 的 `bundle-manifest.json` SHA-256：`7f2fa98421d2718e95fe8f802316e080d3304386fca1c549f07426a6d637a1e3`。这份 payload 在 CPU 检查后保持原样；后续 GPU 对照收据单独保存在 `checks/`，不得据其结果改写 payload、任务或旧成绩。

```powershell
wsl --distribution Ubuntu --exec /mnt/e/CodexRuntime/ReadCue/.venv-linux/bin/python -B -m unittest discover -s /mnt/d/Code/ReadCue/tests -p test_model_research_bundle.py -v
```

本机候选 CPU 完整加载命令（`--output` 必须尚不存在，复跑请改名）：

```powershell
wsl --distribution Ubuntu --exec env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /mnt/e/CodexRuntime/ReadCue/.venv-linux/bin/python -B /mnt/e/CodexRuntime/ReadCue/releases/readcue-v2-candidate-2026-10-28-r2/inference.py check --load-model --base-dir /mnt/e/CodexRuntime/ReadCue/models/qwen3-0.6b --output /mnt/e/CodexRuntime/ReadCue/releases/readcue-v2-candidate-2026-10-28-r2/checks/cpu-load-20261028
```

便携 GPU 冒烟命令：

```powershell
wsl --distribution Ubuntu --exec /mnt/e/CodexRuntime/ReadCue/.venv-linux/bin/python -B /mnt/e/CodexRuntime/ReadCue/releases/readcue-v2-candidate-2026-10-28-r2/inference.py predict --base-dir /mnt/e/CodexRuntime/ReadCue/models/qwen3-0.6b --inputs /mnt/e/CodexRuntime/ReadCue/releases/readcue-v2-candidate-2026-10-28-r2/example-inputs.jsonl --output /mnt/e/CodexRuntime/ReadCue/releases/readcue-v2-candidate-2026-10-28-r2/checks/gpu-smoke-20261028
```

演示输入为另写的一条助手原创 id/text 句子，没有标准输出，不作为新增评测集。与原 runner 的执行等价性应使用相同输入逐条比较生成 token IDs、raw output、prediction 和 status；单纯加载成功不能代替该检查。

已完成 GPU 执行等价冒烟：从219条开发诊断的纯 id/text 输入中按原顺序取前3条，在 `/tmp`（源码 checkout 外）执行候选入口，与冻结 runner 的对应输出比较。3条的生成 token IDs、token 数、原始输出、最终文本、状态和错误字段共18项全部相同，均为 ok。候选推理阶段耗时75.66秒，含加载、不含先行文件哈希；此数字不是单句延迟。收据为 `checks/gpu-parity-20261028/receipt.json` 和 `parity.json`，检查脚本位于诊断目录的 `verify_portable_parity.py`。这是执行一致性检查，不是新增模型效果评测，也不证明任意平台的数值一致性。

交付时仓库完整标准库测试353/353通过、无跳过。研究候选 payload 未因诊断或冒烟结果而更新；本次开发诊断的完整结果另外见[诊断报告](MODEL_DEVELOPMENT_DIAGNOSTIC_V2.md)。

## 研究与发布边界

v2 的既有测量、语义错误和参考覆盖问题见[模型卡](MODEL_CARD_V2.md)与[原运行记录](MODEL_EXPERIMENT_RUN_V2.md)。45 条助手原创合成情境的整句匹配 32/45 不是论文结果、发音准确率、人类验收或自然评论泛化证据。单次训练、评测隔离披露、逐候选仅一名助手审阅等限制继续适用。此次打包不修改参考、原分数、训练、提示或检查点，也不增加 TTS 效果主张。

正式公开前仍须单独确定 adapter 发布条款、公开位置与最终再分发内容；当前没有虚构模型下载地址、发布标签、DOI 或商用保证。代码、基座、训练材料与 adapter 的许可身份分别记录，来源和许可边界见[数据台账](DATA_AND_LICENSES.md)与[发布说明](RELEASING.md)。
