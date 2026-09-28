# 复现中文 pilot

完整模型复现需要 Linux / WSL、Python 3.12、支持 CUDA 12.4 的 NVIDIA 驱动。原始批次使用 RTX 2060 6GB，运行现成基线的本地推理，没有付费 API 或训练。只想检查仓库或运行原样保留基线，可使用 [README 中的轻量流程](../README.md#lightweight-development)。

## 环境

先克隆仓库，在仓库根目录创建独立环境，安装核心包和固定推理依赖。Windows 用户在 WSL 中运行以下命令；确认 `python3 --version` 为 3.12。后续始终使用同一个环境的解释器。

```sh
python3 -m venv .venv-linux
.venv-linux/bin/python -m pip install "setuptools>=77" wheel
.venv-linux/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv-linux/bin/python -m pip install -r requirements-zh-pilot.lock
.venv-linux/bin/python -m readcue verify
```

首次需要下载较大的 CUDA 依赖。锁文件是本次实际选择的 Linux Python 3.12 版本集合，不是跨平台通用环境。

## 基线权重

```sh
.venv-linux/bin/python -m readcue fetch --baseline tn --revision 97f7860d117f651f9aeba1bc6558eec7c378c958 --destination models/zh-tn
.venv-linux/bin/python -m readcue fetch --baseline qwen --revision c1899de289a04d12100db370d81485cdf75e47ca --destination models/qwen3-0.6b
```

下载前先阅读[基线许可](DATA_AND_LICENSES.md#中文本地-pilot2026-09-28)：专用 TN 是非商业研究对照。`fetch` 只获取允许列表内的模型、tokenizer 和说明文件，逐文件核验上游 LFS SHA-256 或 Git blob 哈希，再生成本地资产清单。已有文件必须匹配所选 revision；目录内的冲突文件、未登记内容或链接会导致停止，使用独立的模型目录。具体历史资产哈希见 [freeze.json](../eval/zh_reading_pilot_v1/freeze.json)。不下载或执行训练脚本、pickle 训练参数及远程模型代码。

专用 TN 的导出格式来自 Transformers 5.13.0，本轮固定运行库为 4.57.6。执行脚本仅在内存中把特殊 token 列表字段和默认 RoPE 参数映射到旧版字段，核对完整词表编号、特殊 token 标志及 FIM 提示标记；不修改下载文件。两个模型都显式开启推理 KV cache。首次加载失败及修复时序保存在 freeze.json；此次兼容失败发生在任何神经模型预测之前。

## 推理和评分

以下输出目录必须是新的，程序拒绝覆盖已有预测。依次运行，避免两个 GPU 模型争用显存。

```sh
.venv-linux/bin/python -m readcue run --inputs eval/zh_reading_pilot_v1/inputs.jsonl --baseline identity --output runs/reproduction/identity
.venv-linux/bin/python -m readcue run --inputs eval/zh_reading_pilot_v1/inputs.jsonl --baseline wetext --output runs/reproduction/wetext
.venv-linux/bin/python -m readcue run --inputs eval/zh_reading_pilot_v1/inputs.jsonl --baseline tn --model-dir models/zh-tn --revision 97f7860d117f651f9aeba1bc6558eec7c378c958 --output runs/reproduction/tn
.venv-linux/bin/python -m readcue run --inputs eval/zh_reading_pilot_v1/inputs.jsonl --baseline qwen --model-dir models/qwen3-0.6b --revision c1899de289a04d12100db370d81485cdf75e47ca --output runs/reproduction/qwen
.venv-linux/bin/python -m readcue score --cases eval/zh_reading_pilot_v1/cases.jsonl --predictions runs/reproduction/tn/predictions.jsonl --output runs/reproduction/tn/score.json
```

将最后一条中的 tn 替换为各基线名称，即可逐个评分。每个运行目录保存预测、元数据、日志及 `run.json`；加载失败也会保留失败记录。模型输入只有 id/text；参考、类别和理由只在离线评分时读取。环境与权重加载失败须先处理，不能把没有执行的模型打零分。

原始 `scripts/prepare_zh_models.py`、`scripts/run_zh_baseline.py` 和 `scripts/score_zh_pilot.py` 保留冻结字节用于历史追溯。新运行优先使用上述 CLI：它核验来源和输入，并调用原始推理/评分实现；直接运行旧脚本不能获得全部外围核验与失败记录。原始四组输出位于 [reports/zh-pilot-v1](../reports/zh-pilot-v1/REPORT.md)，不要覆盖。

四种输出完整后，可核对冻结文件、输入/输出哈希和运行记录并归档逐例证据：

```sh
.venv-linux/bin/python -m readcue archive --run runs/reproduction --output reports/reproduction
```

归档入口要求新目录，拒绝覆盖既有报告。原始批次的 identity 早于冻结清单最终时间戳，之后规则配置和神经模型兼容层也有有记录的变化；清单明确保留这些时序，因此不将本轮称为不可变预注册实验。

```sh
.venv-linux/bin/python -m unittest discover -s tests -v
```

这些测试只检查评分、输入完整性与编辑解析风险。模型是否有用由真实预测及后续人工/实际使用验证决定，不由测试通过数决定。
