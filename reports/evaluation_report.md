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

### 口径 L（本地 Qwen3-4B 答题，零 API 成本）

（进行中：V1' 基线重跑 + V3' / V5' 对照，完成后填入）

| 版本 | Overall (500题) | 子集50 | 说明 |
|---|---|---|---|
| V1' 词法笔记（锚点） | 待填 | | |
| V3' 双层记忆 | 待填 | | |
| V5' 融合旗舰 | 待填 | | |

## 3. 成本与效率（手册附录 C.3 要求披露）

### 口径 D 实测（DeepSeek，2026-10-14）

| 项目 | prompt tokens | completion tokens | 说明 |
|---|---|---|---|
| V1 全量答题 | 1,219,063 | 8,666 | 648s（串行） |
| V2 全量答题 | 1,431,564 | 8,628 | 1520s |
| V1+V2 判分×2 | ~800,000 | ~5,000 | 每题 ~1.6k tok |
| 子集/调试/失败重试 | ~600,000 | ~50,000 | 估算 |
| **合计** | **~4.05M** | **~72k** | **账户余额耗尽（402）** |

按 DeepSeek-chat 定价折算约 ¥5-10；教训：**迭代期应默认使用本地零成本后端**（已切换，见 ITERATIONS.md 迭代五）。

### 口径 L 实测（本地 RTX 3060 6GB，Qwen3-4B Q4_K_M，-ngl 99）

（待填：token/题、答题时延、建库时延、检索时延）

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

- 无针对评测题的硬编码、特判或规则匹配；所有策略对 500 题统一生效；
- 未将全量历史拼入上下文（每组证据独立建库，证据窗口 ≤ 36 卡 / 聚合题型 ≤ 全库 ~100 卡，
  且逐题检索证据存档于 records 可审计）；
- grader prompt 引用官方脚本并注明出处；正式得分以项目组 GPT-4o 复跑为准；
- 使用公开数据集，无涉敏数据；AI 辅助开发过程在本迭代文档中留痕。
