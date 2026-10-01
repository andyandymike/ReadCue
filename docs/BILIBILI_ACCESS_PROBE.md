# B 站相关公开语料获取实测

日期：2026-09-29。已实际取得公开发布文件和 HF 分页样本；没有直接爬取 B 站站内，没有登录或申请 gated 数据集，没有训练、模型推理或付费调用。

## 已跑通的路径

| 来源 | 实际取得 | 完整来源的规模／用途 |
| --- | --- | --- |
| [Midsummra/bilibilicomment](https://huggingface.co/datasets/Midsummra/bilibilicomment) | 官方 viewer API 三个位置各100行，共300条，精确唯一文本289条 | viewer 报告4,923,469行；原CSV元数据为478,839,779字节，约479 MB。没有下载全量CSV |
| [FunnySaltyFish 评论对话示例](https://github.com/FunnySaltyFish/bilibili_comments_crawl/tree/aeac0ba2d92703fa26199a03816ff10fbb404a57/example_data) | 全部9个JSON：55条对话链、303次话轮、163种精确文本，其中162种非空 | 来自两个视频、9个根评论线程；共享前缀不能重复计数，保留对话结构但无逐条时间 |
| [linyiLYi/bilibot](https://github.com/linyiLYi/bilibot/tree/c7466edee03e5ba2d7d60d8a233db4bf5b5c8412/data) | `test.jsonl` 744,251字节、1,916行；完整序列精确唯一1,911条 | 是格式化聊天训练材料，不是已核验的1,916条独立原评论；未下载train/valid |

三种获取路径均不需要 B 站账号或视频文件。只能据此确认公开文件可获取，不推导出上游评论全部授权或未来 HF 权重条款已确定。

## 优先继续检查的评论库

`Midsummra/bilibilicomment` 固定 revision 为 `48ccfb3a7dc6f6f03dff8e56443c067c8bec718f`，公开、非 gated。本轮 offsets 为0、100000、1000000，各请求100行；三次响应均无所选行截断，`X-Revision` 与保存的元数据、README版本一致。这是验证可获取性的三个固定窗口，不是随机或代表性样本。

可见字段只有 `message/time/timestamp`，保留中英混写、重复字符、表情和数字。缺少视频ID、评论ID及明确父子关系，无法仅靠这些字段可靠按视频切分训练和测试，也不能逐条回查原帖。发布者称部分“回复”行对应前一条，这仅是上游说明，没有在本站原始记录上独立核验。本轮派生样本未据此伪造回复关系。

发布者声明数据采自2023年热门视频、未清洗，且提醒游戏区内容偏差；这不是我们独立核准的全库组成。许可元数据标注AGPL-3.0，但README未提供逐条原评论授权证据或未来模型权重的明确条款。可以先做本地来源质量诊断，不能直接承诺采用该库后发布宽松商用权重。

## 其他两份材料的边界

评论示例固定 revision `aeac0ba2d92703fa26199a03816ff10fbb404a57`，JSON及README共69,151字节，逐文件核对Git blob SHA-1及SHA-256。派生对话省去 `from` 作者字段；原 `value` 文本不纠错、不展开缩写、不删除内部标识，转换记录单独保存。README写明仅用于学习交流、不可商业使用；没有确认独立数据许可证。不能将两个视频的示例当作当前全站分布。

bilibot 固定 revision `c7466edee03e5ba2d7d60d8a233db4bf5b5c8412`。每行都包含ChatML的system/user/assistant三段，user去重1,907种、assistant去重1,890种；作者仅说明用 B 站评论微调，未明确哪一段是采集原文、哪一段是构造问题或回答。故不把两种角色合并为真人评论，也不把回答当TN读法标签。仓库Apache-2.0不单独解决评论原作者权利问题。该test文件已经查看，不是 ReadCue 未见评估集。

## 保存与后续

所有原始文件和逐条数据保存在 `E:\CodexRuntime\ReadCue\data\bilibili-access-probe-2026-09-29`，不进入Git。

- `hf_comments/`：元数据、固定README、原始分页响应、300条派生记录、请求时间与哈希。
- `comment_examples/`：9个原始JSON、完整对话与精确去重文本、固定版本和许可声明、采集源码。
- `additional/`：bilibot原始test、README、LICENSE、下载回执和字段审计。
- `README.md`：本地总览与少量原文例子；`file-manifest.json`：本轮文件清单和哈希。

下一步可以先筛查这300条评论的真实语言现象，并与保留了楼中楼的示例互补。全量CSV下载在技术上有明确公开入口、大小和上游LFS哈希，尚未下载也尚未验证整包哈希。是否扩大获取应取决于样本质量、划分可行性和发布所需的授权证据。
