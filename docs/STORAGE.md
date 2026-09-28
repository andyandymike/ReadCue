# 本地存储与缓存恢复

2026-09-28 的只读审计共计 **12,407,400,305 字节，约 11.56 GiB**。这是普通文件的逻辑大小，不是 NTFS 分配空间；跳过了 4 个 WSL 链接，没有进入其目标。硬链接若出现会按路径计数；正在变化的文件会影响扫描结果。

| 内容 | 审计大小 | 用途 |
|---|---:|---|
| `.venv-linux/` | 5.90 GiB | 已安装的 Python、PyTorch、CUDA、Triton、Pynini 等运行依赖 |
| `.cache/` | 3.11 GiB | 主要是已经安装过的依赖下载包 |
| `models/` | 2.54 GiB | Qwen3-0.6B 与中文 TN 两份不同的基线模型 |
| 源码、报告、实验记录及 Git 等 | 约 0.91 MiB | 项目和实验依据 |

日常修改代码、检查评分逻辑和运行轻量测试，无需先安装 GPU 依赖。完整 GPU 环境只用于本地神经模型推理，具体步骤见 [中文 pilot 复现说明](REPRODUCING_ZH_PILOT.md)。`.venv-linux/` 和 `models/` 暂时保留，避免每轮评测重新安装和下载。不要从已安装环境中手工删除 CUDA 动态库。

## 只读查看占用

从仓库根目录使用系统 Python 即可，不需要项目虚拟环境：

```sh
python tools/storage_audit.py
python tools/storage_audit.py --json
```

工具只向终端输出，按一级目录汇总普通文件的逻辑字节数，不写报告文件；跳过 symlink、junction 和其他 reparse point，并列出未读取路径和错误。`--root` 可以显式指定待审计目录。

## 可重建的 wheel 下载缓存

`.cache/linux-wheels/` 原有 **43 个 wheel，共 3,268,170,375 字节（3.044 GiB）**。它们是已安装依赖的压缩下载包，与 `.venv-linux/` 中的运行文件用途不同。审计逐一确认了 SHA-256，并确认 43 个对应安装包的版本一致；没有额外或缺失的 wheel。

[固定恢复清单](maintenance/2026-09-28-wheel-cache.json) 保存这 43 个文件的路径、大小、修改时间、SHA-256、包版本及官方 PyPI 下载 URL，并记录旧 `.cache/linux-resolve.json` 的原始 SHA-256。保留该清单及恢复工具后，安装完成的 wheel 缓存可以按清单清理；网络源将来可能不可用，清单不等于离线备份。

恢复工具默认仅检查文件和列出计划，**不联网、不写文件**：

```sh
python tools/restore_wheel_cache.py
python tools/restore_wheel_cache.py --package colorama
```

明确添加 `--download` 才会下载缺少的文件；可以只恢复一个小包检查下载流程：

```sh
python tools/restore_wheel_cache.py --package colorama --download
python tools/restore_wheel_cache.py --download
```

下载只接受清单中的 `https://files.pythonhosted.org/packages/` URL，流式检查大小及 SHA-256，核验成功后才发布到最终路径。已有正确文件直接跳过；遇到冲突、链接路径或校验错误就停止，不覆盖文件。临时下载失败会移除本次临时文件。恢复工具只负责这 43 个 wheel，不安装依赖，也不操作模型。

`.cache/wetext/` 的两份规则缓存仅约 1.21 MiB，正式基线仍会读取，当前保留。`uv` 自举文件约 71.34 MiB，不在这次 43 个 wheel 的清单范围内。

## 清理记录

2026-09-28 已清理清单中的43个已安装依赖下载包，删除前再次逐个核对路径、大小、SHA-256，并确认没有下载或安装进程使用它们。随后从官方地址重新下载 colorama 小包并验证哈希，保留该25,335字节文件作为恢复抽查；因此净回收 **3,268,145,040字节，约3.04 GiB**。

紧接清理后的逻辑大小为 **9,139,354,918字节，约8.51 GiB**；这次扫描仍跳过4个链接，无读取错误。后续核心开发环境、源码和Git初始化会有少量新增，不应将其他进程导致的磁盘空闲变化等同于此次回收量。[执行记录](maintenance/2026-09-28-storage-cleanup.json)保留文件清单、恢复抽查与清理后扫描结果。

全部冻结源码、两份模型的所有记录资产、四组归档预测重新核对原始哈希通过；`.venv-linux/` 和模型大小未变。空间最大的剩余部分是实际运行依赖与权重，并非重复缓存。

只处理清单中的确切文件。不要用整目录 `git clean` 清除忽略文件：模型、运行环境、未归档实验输出也被 Git 忽略。原始实验记录很小，应予保留。
