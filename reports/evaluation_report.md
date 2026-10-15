# LongMemEval Oracle 评估报告（迭代中 · 持续更新）

> 项目：长交互式AI记忆系统（D.I.V.E. × 川大 · 个人赛）
> 基准：LongMemEval Oracle 档 500 题（`data/longmemeval/longmemeval_oracle.json`，15.4MB，
> 来源 `hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned`，ICLR 2025 论文官方数据）
> 数据版本：2026-10-14 下载；题目分布：temporal-reasoning 133 / multi-session 133 /
> knowledge-update 78 / single-session-user 70 / single-session-assistant 56 / single-session-preference 30
> （其中 30 题为 `_abs` 拒答题，按 question_id 后缀标识）

## 1. 评测协议

- **流水线**：按干草堆分组建库（同一证据会话组只 ingest 一次）→ 检索 → 生成 → JSONL
  （每行 `{"question_id", "hypothesis"}`）→ 判分。
- **判分 prompt**：逐字引用官方 `src/evaluation/evaluate_qa.py`（5 类题型模板 + abstention 模板，
  见 `systems/judge_official.py` 头部出处声明）。temperature=0，max_tokens=10，回答 yes/no。
- **双口径说明**：
  - 口径 D（DeepSeek）：答题 = DeepSeek-chat；判分 = 官方 prompt + DeepSeek-chat。
    **非官方 judge，仅用于迭代相对比较**；
  - 口径 L（Local）：答题 = Qwen3-4B-Instruct-2507（本地 llama.cpp，GPU 全量 offload）；
    判分 = 官方 prompt + 同款本地模型。
  - 正式 Benchmark 得分由项目组以 GPT-4o judge 统一复跑（手册附录 C.3），自报分不作为计分依据。
- **可比性纪律**：版本间对比只在同口径内进行；口径切换（D→L）后重跑 V1 基线作为锚点。
- **可复现**：50 题快评 = 分层等比抽样（seed=42，`stratified_subset`）；全量 = 500 题。
  所有 runs 产物在 `runs/<system>/`：hyp JSONL + 逐题 records（含检索分数、证据数、拒答标记）。

## 2. 结果总览

### 口径 D（DeepSeek-chat 答题）

| 版本 | Overall (500题) | 子集50 | 说明 |
|---|---|---|---|
| V1.2 词法笔记 | **0.7140** | 0.660 | 提交基线 |
| V2 稠密混合 | 0.6960 | 0.660 | 稠密通道为负贡献（见 §4） |
| V3 双层记忆 | —（API 耗尽） | 0.700 | 子集口径优于 V1.2 同子集 +0.04 |
| V4 路由+恢复 | —（API 耗尽） | **0.760** | 拒答恢复 6 例，reasoner 路由 |
| V5 融合 | — | 待跑 | 本地口径进行中 |

**V1 全量分题型**（judge=DeepSeek 复判，n=500）：

| 题型 | Acc | n |
|---|---|---|
| single-session-user | 0.9375 | 64 |
| single-session-user-abs | 1.0000 | 6 |
| knowledge-update | 0.8611 | 72 |
| knowledge-update-abs | 1.0000 | 6 |
| temporal-reasoning | 0.7244 | 127 |
| temporal-reasoning-abs | 0.8333 | 6 |
| multi-session | 0.5455 | 121 |
| multi-session-abs | 0.9167 | 12 |
| single-session-assistant | 0.5000 | 56 |
| single-session-preference | 0.7000 | 30 |
| **Overall** | **0.7140** | 500 |

### 口径 Z（智谱 coding plan：glm-5.3-flash 答题/判分，glm-5.3 难题路由；2026-10-14，召回预算策略）

**召回预算合规**：所有题型一律按检索排序取预算内证据（普通题 top-24，聚合题 top-48），
单题证据均值 ~1.65k token（p95 3.2k、max 3.7k），对原始干草堆压缩比 4:1；
任何题型都不拼塞全库/全历史（手册 F.1 合规，见 §5）。每题证据量逐题存档于 records 可审计。

| 版本 | Overall (500题) | 相对 V1 | 备注 |
|---|---|---|---|
| V1 词法笔记（锚点） | 0.7820 | — | 20.1 min，1.01M in + 103k out |
| V2 稠密混合 | 0.8120 | +3.0pt | temporal 0.819→0.898；GLM 口径下稠密通道有效 |
| V3 双层记忆 | 0.8400 | +5.8pt | ss-assistant 0.50→**0.946**；multi-session 0.793 |
| V4 路由+恢复 | 0.8200 | +3.8pt | multi-session 0.736→**0.868**；代价：_abs 拒答题被误捞（ms-abs 0.833→0.583） |
| **V5 融合旗舰** | **0.8980** | **+11.6pt** | V3 写入 × V4 读取，18 例假拒答恢复；temporal **0.906**、multi-session **0.893** |

**V5 分题型（口径 Z，n=500）**：ss-user 0.953 ｜ ss-user-abs 1.000 ｜ knowledge-update 0.958 ｜
knowledge-update-abs 0.500 ｜ temporal 0.906 ｜ temporal-abs 0.833 ｜ multi-session 0.893 ｜
multi-session-abs 0.750 ｜ ss-assistant 0.911 ｜ ss-preference 0.733 ｜ **Overall 0.8980**

**机制贡献归因（500 题口径）**：稠密混合检索 +3.0 ｜ 双层记忆 +5.8（其中助手笔记切块对
ss-assistant 单题型 +44.6pt）｜ 题型路由+拒答恢复 +3.8（对 V1）/ V5 组合再 +5.8（对 V3）——
写入端与读取端机制正交且可叠加。

**V5 交付物**：`deliverables/lmem_oracle_v5_zhipu.jsonl`（500 行，`question_id`+`hypothesis`）。

**预算策略的代价（诚实披露）**：聚合题从"全库读取"改为 top-48 后，V1 从 0.8000 降至 0.7820
（multi-session 0.760→0.736）。这是红线合规的代价；检索排序召回在 96% 的题目上无损，
损失集中在极少数需要全库枚举的计数题。

### 口径 L（本地 Qwen3-4B 答题，零 API 成本）

V1' 子集50 = 0.600（4B 答题 + 4B judge）。本地后端保留为零成本迭代设施与消融基准；
正式对比在口径 Z 内进行。

## 3. 成本与效率（手册附录 C.3 要求披露）

### 口径 Z 成本（2026-10-14，glm-5.3-flash / glm-5.3 via coding plan）

| 系统 | 答题耗时 | prompt tokens | completion tokens |
|---|---|---|---|
| V1 全量 | 1205s | 1,009,378 | 103,406 |
| V2 全量 | 1975s | 1,291,497 | 106,030 |
| V3 全量（含抽取） | 4910s | 1,204,084 | 108,064 |
| V4 全量 | 3619s | 1,045,723 | 144,226 |
| V5 全量（抽取走缓存） | 2673s | 1,225,668 | 140,631 |
| 判分 ×5（官方 prompt，~0.5k tok/题） | — | ~1,250,000 | ~30,000 |
| 子集/调试/失败重试 | — | ~1,500,000 | ~200,000 |
| **合计** | — | **~7.5M** | **~0.73M** |

检索效率：单题证据均值 6.6k 字符（≈1.65k token）、p95 12.9k、max 14.8k；
BM25/RRF 检索单题 < 5ms（进程内索引）；建库 V1/V2 即时（规则）、V3/V5 约 25 min（LLM 抽取，有磁盘缓存可复用）。

### 口径 D 成本（已停用）

DeepSeek 合计 ~4M tokens（约 ¥5-10）后余额耗尽（402），V3/V4/V5 迁移至口径 Z。
教训：**迭代期默认用零成本本地后端或有配额保障的端点，API 只用于终版快照**。

## 4. 失败模式分析（驱动迭代的证据链）

1. **V1 全量 143 失败解剖**：107/143（75%）为**假拒答**（检索分数不低、答案在库、模型弃权）；
   真答错 34；拒答题仅丢 2/30。→ 直接催生 V4 的 two-pass 拒答恢复（子集 6 例捞回）。
2. **检索均值对比**：失败题 top_bm25 均值 12.0 vs 正确题 13.7 —— 检索信号区分度低，
   佐证"瓶颈在读取端而非召回端"（与 V2 稠密检索无净增益相互印证）。
3. **V2 负结果**：oracle 档干草堆小（每组 ~100 卡），BM25 词面覆盖已高；稠密近邻引入
   主题相关但不含答案的卡片稀释精确度。该架构对 S/M 档（大干草堆）预期有效，保留。
4. **V3 故障教训**：纯 LLM 原子层（无情景保底）子集暴跌至 0.440 —— 抽取丢失 +
   JSON 校验静默失败。修复 = 宽松解析 + **双层写入**（语义原子层 + 情景原文层），
   修复后 0.700。印证手册"分层记忆"设计价值。
5. **淘汰策略 bug**：乘法打分在 access=0 时坍缩 → 改加性混合（单元测试覆盖：
   `tests/test_v3_forget.py`，版本化/as-of/去重/容量淘汰）。

## 5. 诚信与合规声明

- **无上下文拼塞**：任何题型均通过记忆检索取预算内证据（普通 top-24 / 聚合 top-48），
  单题证据均值 ~1.65k token（max 3.7k），原始干草堆均值 26.4k 字符（max 109k），
  压缩比 4:1；与底层 LLM 的上下文窗口大小无关，更换长窗口模型不改变本系统行为；
- **无针对评测题的硬编码、特判或规则匹配**；所有策略对 500 题统一生效，
  逐题检索证据（分数、证据数、证据字符数、拒答标记）存档于 records 可审计；
- grader prompt 引用官方脚本并注明出处；正式得分以项目组 GPT-4o 复跑为准；
- 使用公开数据集，无涉敏数据；AI 辅助开发过程在本迭代文档中留痕。
