# 2026-10-01 环境与存储优化

本轮针对默认安装副本落后于源码、训练环境隐式借用推理环境、模型入口未统一约束缓存位置，以及验证残留长期累积的问题。没有训练新权重、更改数据或重算首轮评测。当前目录与日常使用规则见[存储说明](../STORAGE.md)。

## 已完成的调整

- Windows `.venv` 与 WSL `.venv-linux` 都以 editable 方式加载当前 checkout；在独立工作目录、没有 `PYTHONPATH` 的进程中检查实际导入位置与 CLI。
- 将固定训练依赖离线合并到 `E:/CodexRuntime/ReadCue/.venv-linux`。原 43 个推理依赖的版本、位置和 RECORD 哈希保持一致，所有模型锁定版本匹配，两个环境的依赖完整性检查通过。
- 旧 `.venv-train-v1` 路径改为指向统一 GPU 环境的 junction。兼容入口实测 CUDA 可用，加载当前源码，已无原跨环境 `.pth`。原训练 overlay 已迁到本轮维护目录等待清理，不再是运行依赖。
- 模型入口核对实际加载源码与 `--project`，记录入口、模块路径和哈希；加载 ML 库前检查全部锁定版本，拒绝缺失或漂移。
- 明确要求共享 `--budget-root`，统一 HF、Torch、Triton、CUDA 缓存与阶段临时目录，纳入原 5 GiB 检查。独立进程、缓存归属、链接及无法读取路径的约束见存储说明。模块入口需 `-B` 或启动前禁写字节码。

本次执行源码与首轮训练的冻结源码不同，两者分别保留。固定基座、任务配置与训练/预测计算未变；新维护检查不能冒充首轮比较的重算。

## 清理与尚未完成的删除

已按确切路径清除四轮完成用途的 11 个验证环境／解包目录，共 112,743,142 字节；保留各轮发行包、日志和验证收据。同时删除仓库 `.cache` 下的 uv 二进制、已验证 wheel 和截断 tar 三份残留，共 74,800,149 字节。合计 **187,543,291 字节，178.86 MiB**。上述大小是被删除普通文件的逻辑字节数，不等同于卷空闲空间增量。

两份旧备份已经逐文件重新核对 SHA-256、使用情况与原收据；用户明确授权永久删除，但自动审批在执行前拒绝，返回仅为 `blocked by policy`。当前仍存在，删除量为零：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `E:/CodexRuntime/ReadCue/maintenance/wsl-2026-09-28/ubuntu-before-compaction.vhdx` | 10,599,006,208 | `0d9596fd1b44d1a602b22ca7dc9ab0d512c72d6406e08470ba19536efdf2e079` |
| `E:/CodexRuntime/ReadCue/archives/cleanup-2026-09-28/install-residuals-13files.tar` | 2,377,820,160 | `25c17e99cf2bd8a7a608698790b2e49ddd852e0a788fd4a7ca2108c8460997a3` |

合计 12,976,826,368 字节（12.09 GiB）。当前 Ubuntu 注册磁盘另有路径，本轮没有操作它，也没有复制新的大恢复包。删除退休 overlay 前发现其中 `lib64 -> lib` 链接，检查停止；随后单独移除该链接的命令也被自动审批拒绝，退休 overlay（约 40.1 MiB）仍在维护目录。本轮不绕过这些拒绝。

维护收尾会继续清理本轮新建的可重建验证副本，精确完成状态以本机 `verification-cleanup-result.json` 为准，不补写历史删除收据。模型、原始数据、人工审阅版本、源码快照、训练和评测依据保留。

## 验证及本机证据

新增边界的定向测试在 Windows 与 Linux 均运行 23 项通过，Windows 因符号链接权限跳过 1 项，Linux 实际执行通过。完整测试日志分别保存为 `logs/full-tests-windows.log` 和 `logs/full-tests-linux.log`；维护完成收据记录最终状态。独立检查也确认源码身份、锁版本与缓存归属逻辑，模块入口的提前字节码写入限制已写入启动要求。

统一 GPU 环境使用新的公开入口、独立预算根和首轮选定 adapter，对一条无标签维护样例实际加载并预测成功：输入“缓存与目录检查已经完成。”，输出保留原句。没有加载评测参考、更新权重或计算效果分数。这只证明当前环境和入口可运行，不证明读法或 TTS 质量提升。

7 份原 pilot 冻结文件、两份基座模型的登记资产、首轮完成收据绑定的 22 份实验依据及选定 adapter SHA-256 重新核对通过。源码发行验证另用新的隔离目录进行 wheel/sdist 构建与离线安装，保留源码逐字节比较和无需 ML 依赖的核心导入检查；本轮没有公开发布。

全部新证据位于本机 `E:/CodexRuntime/ReadCue/maintenance/runtime-optimization-2026-10-01/`，该路径不是公开下载地址：

| 记录 | 用途 |
|---|---|
| `environment-consolidation.json`、`receipts/*environment*.json` | 环境来源、合并及兼容入口探针 |
| `receipts/linux-original-dependencies-preserved.json` | 原推理依赖未变证明 |
| `temporary-cleanup-manifest.json`、`temporary-cleanup-result.json` | 已完成的 14 个清理目标、逐文件哈希及删除量 |
| `backup-retirement-manifest.json`、`backup-retirement-result.json` | 两份旧备份核验及工具策略拒绝记录 |
| `retired-environment-result.json` | 退休环境仍被保留的原因 |
| `model-smoke-experiment/predict/` | 实际模型加载、预测、源码及路径身份 |
| `receipts/protected-artifacts-after.json` | 冻结材料、资产和原实验依据复核 |
| `verification/final-package-verification.json` | 最终源码包与 wheel 的独立安装验证 |
| `completion-receipt.json` | 本轮最终验证与清理状态 |
