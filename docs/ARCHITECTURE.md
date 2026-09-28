# 代码与实验边界

ReadCue 目前是小规模研究项目，没有训练框架或产品服务。可维护性首先意味着实验身份、数据来源和失败记录可靠；不为尚未确定的模型方向搭插件系统、训练平台或多层抽象。

## 两条代码路径

**历史复现路径固定不动。** `eval/zh_reading_pilot_v1/freeze.json` 记录三份脚本、协议和样例的哈希。`scripts/run_zh_baseline.py`、`scripts/score_zh_pilot.py`、`scripts/prepare_zh_models.py` 保留原始字节。已有预测、元数据和评分不因代码整理而覆盖。不要修改旧哈希来让一次重构通过校验。

**新增工程入口可以演进。** `src/readcue` 提供零强制运行依赖的工具包，入口是 `python -m readcue` 或 `readcue`。命令接受 `--project` 指定项目 checkout，默认当前目录。安装的包不夹带模型、数据或历史报告；评分和复现仍需该 checkout。

| 文件 | 职责 |
| --- | --- |
| `src/readcue/artifacts.py` | 有界内存哈希、上游 Git blob / LFS 校验、校验后发布下载文件、原子 JSON 记录、模型身份及冻结文件核验 |
| `src/readcue/evaluation.py` | 校验后加载 v1 原始评分器；校验每个系统身份、revision和结果哈希，再归档 |
| `src/readcue/baselines.py` | 严格 id/text 输入、模型资产预检、子进程调用历史 runner、持久状态和错误日志 |
| `src/readcue/cli.py` | 薄命令入口，组织 verify / fetch / run / score / archive |
| `scripts/archive_zh_pilot.py` | 未被旧冻结清单绑定的兼容入口，转到新归档实现 |

评分不复制第二套实现；新包核对旧评分器及协议后再调用它。神经推理也不复制，继续使用已审查的历史 runner。模型与 WeText 库只在明确执行相应 baseline 的子进程中加载。新架构不是一个训练结果，不改变首轮研究结论。

## 本地使用

核心安装不下载 Torch、CUDA、模型或规则包：

```sh
python -m pip install --no-deps --no-build-isolation -e .
python -m readcue verify
python -m readcue score --cases eval/zh_reading_pilot_v1/cases.jsonl --predictions reports/zh-pilot-v1/tn.jsonl --output runs/inspection/tn.score.json
```

上述安装需要已有 setuptools>=68 和 wheel。历史 GPU / WeText 环境仍按 [复现说明](REPRODUCING_ZH_PILOT.md)的独立锁文件准备；`requirements-zh-pilot.lock` 是那次 Linux 环境记录，不是所有使用者都要安装的核心依赖。

新运行使用一个全新目录，即使导入、模型加载或预热失败，也会留下 `run.json` 和已产生的日志。不会把未运行的模型当作能力零分：

```sh
python -m readcue run --inputs eval/zh_reading_pilot_v1/inputs.jsonl --baseline identity --output runs/new-pilot/identity
python -m readcue fetch --baseline tn --revision 97f7860d117f651f9aeba1bc6558eec7c378c958 --destination models/zh-tn
python -m readcue run --inputs eval/zh_reading_pilot_v1/inputs.jsonl --baseline tn --model-dir models/zh-tn --revision 97f7860d117f651f9aeba1bc6558eec7c378c958 --output runs/new-pilot/tn
```

`fetch` 会访问网络并下载资产；`run` 不自动安装依赖或下载模型。示例只是使用说明，不表示已重新运行神经模型。运行前核对模型 repo、revision、每个文件内容；历史缓存可与冻结清单中的逐文件 SHA-256 对照，新下载则对每个文件使用 HF 返回的 LFS SHA-256 或 Git blob SHA-1 来源证明。Git blob 哈希不是普通文件 SHA-1，包含长度头。

模型目录只允许清单登记的普通文件和 `readcue_manifest.json`。额外的 `chat_template.jinja`、adapter、子目录、符号链接或 Windows 重解析文件都被拒绝，避免加载器读到未核验的覆盖配置。下载也拒绝这类残留内容；工具不会替用户删除它们，应使用独立的干净资产目录。

四个系统都运行完后，归档接受 `runs/new-pilot/<baseline>/predictions.jsonl` 这种新目录，也接受旧版同目录四组命名文件：

```sh
python -m readcue archive --run runs/new-pilot --output reports/new-pilot
```

该归档命令只接受原始 pilot 的输入哈希和四个已记录的基线身份；新样例可单独调用 `score`，不能冒充原始 pilot。原始预测可以包含逐例失败，但未完成的外层运行不能伪装为完整归档。归档写入失败会留下明确失败状态，不覆盖旧报告。

## 扩展方式

下一次实验先新增独立案例与协议，不修改 v1。当前输入/预测 JSONL 和逐句失败字段已经足够，不急于建立通用数据框架。真实需求确定后，再增加一个明确后端或独立实验 runner；保留旧版本并给新来源、配置、提示和评分建立独立清单。

目前没有抽象模型注册中心、插件发现、训练调度、自动发布或自动 GPU 申请。是否训练、花费多少以及数据能否发布仍按项目计划决定。

## 验证

CI 不安装模型依赖，也不联网下载或运行神经模型。新增测试用已有四组共 288 条预测逐项重算，要求所有历史评分和汇总不变；另覆盖系统/revision错标、gold字段和重复id拒绝、子进程失败持久记录、伪网络的 Git/LFS 校验、校验失败不发布文件，以及损坏缓存不覆盖。identity 子进程检查只返回原文。

```sh
python -m unittest discover -s tests -v
```

已有七项评分/解析测试继续保留。新包的任何整理都不应通过“更新参考、覆盖旧预测、改写冻结哈希”获得通过。
