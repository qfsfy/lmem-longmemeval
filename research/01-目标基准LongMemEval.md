# 目标基准确认：LongMemEval（500 题）

> 调研日期：2026-09-21 ｜ 结论：**"500 题的题库" = LongMemEval**（此前记忆中的名字已找回）
>
> **⚠️ 2026-09-22 修订**：学生手册（research/07）确认本项目实际评测 **Oracle 档**（`longmemeval_oracle.json`，500 题，历史已剪至证据会话），并**禁止**全量拼上下文的投机实现。本文 S/M 档的描述保留作背景知识，评测协议（judge、JSONL 格式）与 Oracle 档一致。

## 1. 基本信息

| 项 | 内容 |
|---|---|
| 论文 | *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*（Wu et al., ICLR 2025） |
| arXiv | https://arxiv.org/abs/2410.10813 |
| 官方仓库 | https://github.com/xiaowu0162/LongMemEval （MIT License） |
| 数据 | HuggingFace `xiaowu0162/LongMemEval`（2025/09 有清理版更新） |
| 出品方 | Tencent AI Lab / UCLA |
| 后续 | LongMemEval-V2 已于 2026/05 宣布（面向 agentic-context memory），值得关注 |

## 2. 基准构成

- **500 道高质量题目**，以"大海捞针（needle-in-a-haystack）"方式构造：每题挂载一段带时间戳的多会话聊天历史。
- **两个规模版本**：
  - `LongMemEval_S`：历史约 **115K tokens**（约 40 个会话）——先用它做开发；
  - `LongMemEval_M`：历史约 **500 个会话（约 500K tokens）**——最终目标。
- 另有 **oracle 版本**（只含证据会话），用于定位"检索失败 vs 推理失败"。

## 3. 五大核心能力（与幻觉直接相关）

| 能力 | 题型细分 | 对记忆系统的要求 |
|---|---|---|
| 信息抽取 | 单会话-用户 / 单会话-助手 / 用户偏好 | 精准检索 + 属性级定位 |
| 多会话推理 | — | 跨会话多跳证据聚合 |
| 时间推理 | — | 时间戳元数据、时序索引 |
| 知识更新 | — | 旧记忆失效/覆盖（版本化） |
| **拒答（Abstention）** | 30 题，ID 带 `_abs` 后缀 | 证据不足时拒绝作答——**反幻觉核心题型** |

> 注：信息抽取含 3 个子类型，故有的资料写"6 种题型"，本质是上表 5 大能力。

## 4. 数据字段（每条样例）

`question_id`、`question_type`、`question`、`answer`、`question_date`、
`haystack_session_ids / haystack_dates / haystack_sessions`、`answer_session_ids`；
证据轮次带 `has_answer: true` 标记，可做 **turn 级召回率**评估。

## 5. 官方提供的基线方法（可直接复用代码）

- **长上下文全量**：full-history prompting；
- **检索**：flat-BM25、Contriever、Stella v5 1.5B、GTE-Qwen2-7B；
- **索引/键扩展**：会话摘要、关键词、用户事实画像（separate / merge / replace 三种拼接模式）；
- **时间感知查询扩展与剪枝**；
- 生成端支持本地 **vLLM** 或 OpenAI API。

## 6. 评测协议

1. 把带时间戳的历史喂给自己的记忆系统，输出 JSONL：`{"question_id": ..., "hypothesis": ...}`；
2. 用 `evaluate_qa.py` + GPT-4o 作裁判判分；
3. 用 `print_qa_metrics.py` 汇总指标（QA 准确率、turn 级召回）。
4. `sample_haystack_and_timestamp.py` 可编译任意长度的自定义历史（便于自建中文场景）。

## 7. 关键结论（动机）

- 长上下文模型与商用助手（ChatGPT / Claude 等）在该基准上准确率大致停在 **50–60% 量级**，远未饱和；
- 简单 RAG 式记忆增强并未稳定超越全上下文——**说明"怎么记、怎么检索、怎么拒答"仍有大量研究空间**；
- 各家厂商自报的分数（如 Mem0 称 LOCOMO 92.5 / LongMemEval 94.4）存在争议，复现对比时要以官方 pipeline 为准。

## 8. 相关/衍生基准

| 基准 | 说明 | 链接 |
|---|---|---|
| **LOCOMO**（ACL 2024） | 10 段超长对话（约 300 轮 / 9K tokens，最多 35 会话），QA + 事件摘要 + 多模态 | https://github.com/snap-research/locomo |
| MSC | Multi-Session Chat，早期多会话个性化对话数据 | 见论文 *Beyond Goldfish Memory* |
| PerLTQA | 个人长时记忆问答（记忆分类/检索/融合） | arXiv:2402.16288 |
| MemBench / MemoryAgentBench | 2025 年出现的 agent 记忆专项评测 | 按名检索 |
| **mem0ai/memory-benchmarks** | Mem0 开源的统一评测套件，内置 LongMemEval 跑批 | https://github.com/mem0ai/memory-benchmarks |
| LoCoMo-Plus | 2026 衍生版，加入超越事实型认知记忆评测 | 按名检索 |
