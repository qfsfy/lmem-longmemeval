# 长交互式AI记忆系统（Long-term Interactive AI Memory System）

> D.I.V.E.（赛诺菲 × 四川大学）个人赛交付 ｜ LongMemEval Oracle 统一基准评测
> 立项 2026-09-21 ｜ 当前：多版本迭代评测阶段

一套**与底层 LLM 解耦**的记忆中间层，覆盖记忆五能力（积累 / 更新 / 检索 / 复用 / 遗忘），
以 **5 个渐进版本**实现并通过 **LongMemEval Oracle 500 题**统一基准做可复现评估，
每次迭代只改一个变量，逐版本归因增益（详见 [ITERATIONS.md](ITERATIONS.md)）。

## 版本总览

| 版本 | 一句话 | 写入 | 检索 | 读取 | 状态 |
|---|---|---|---|---|---|
| V1 词法笔记 | 零 LLM 的透明基线 | 规则信息句（寒暄过滤/身份归一） | 自研 BM25 + 画像直取 | LLM 短答案 + 计数协议 | 全量 **0.782**ᶻ |
| V2 稠密语义 | +向量混合检索 | 同 V1 | BM25 ∥ bge-small 余弦 → RRF | 同 V1 | 全量 0.812ᶻ |
| V3 分层记忆 | 语义原子 + 情景保底 | LLM 记忆原子 + 规则句双层 | 混合 + 原子引文去重 | 时序/聚合读取协议 | 全量 0.840ᶻ |
| V4 路由+恢复 | 读取端两杠杆 | 同 V1 | 同 V1 | 题型路由旗舰 + 假拒答 two-pass 捞回 | 全量 0.820ᶻ |
| V5 融合旗舰 | V3 × V4 | 双层原子 | 混合 + 去重 | 路由 + 恢复 | 全量 **0.898**ᶻ |
| V6 优化尝试 | 核验 + 偏好个性化 | 同 V5 | 同 V5（V6.0 曾试 embedding-3） | 路由 + 恢复 + 证据核验 | V6.0 0.838 / V6.1 0.892 / V6.2 0.894ᶻ——未超 V5，**交付 V5** |

ᶻ = 智谱口径（glm-5.3-flash 答题/判分，难题路由 glm-5.3），官方 grader prompt；
另有口径D（DeepSeek）快照：V1 0.714 / V2 0.696。正式得分由项目组以 GPT-4o judge 统一复跑。

## 快速开始

> **环境说明**：本仓库的评测流水线只依赖轻量 Python 包，**无需 conda**——官方
> LongMemEval 仓库的 conda 环境仅在复现其 RAG/长上下文基线时需要；本仓库只使用其
> 数据集（MIT 许可，15MB）与判分协议（grader prompt，见诚信声明）。CUDA 仅在启用
> 本地 llama.cpp 备用后端时用到（RTX 6GB 即可）。

```bash
# 0) 环境（Python >= 3.10；国内建议阿里云镜像 pip install -i https://mirrors.aliyun.com/pypi/simple/）
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# 1) 数据（hf-mirror 下载 Oracle 档，15MB）
curl -L -o data/longmemeval/longmemeval_oracle.json \
  "https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"

# 2a) API 口径：配置 .env（见 .env.example：ZAI_CODING_KEY / DEEPSEEK_API_KEY / ZHIPU_EMBED_KEY）
.venv/Scripts/python run.py eval --system v5 --subset 50      # 50 题分层快评
.venv/Scripts/python run.py judge runs/v5/hyp_v5_sub50_zh.jsonl  # 官方口径判分
# 全量: --system v5 (去掉 --subset)；嵌入后端可选 EMBED_BACKEND=fastembed|zhipu|onnx|onnx_sub

# 2b) 本地口径（零 API 成本）：启动 llama-server 后加 --backend local
scripts/start_local_server.bat
.venv/Scripts/python run.py eval --system v5 --backend local

# 3) 交互体验（含 Mock 离线模式）
.venv/Scripts/python run.py test                                # 离线冒烟测试
.venv/Scripts/python -m lmem.cli ui
```

## 架构（共享平台 + 可替换策略）

```
src/lmem/                    # 平台层（各版本共享）
├── memory/schema.py         # 记忆卡：双时间轴 valid_from/to + observed_at，失效不删除
├── memory/store.py          # SQLite + 进程内 BM25（Okapi，中英混合分词+轻词干化）
├── llm/client.py            # OpenAI 兼容后端（DeepSeek/GLM/本地 llama-server）+ Mock
├── evals/                   # Oracle 适配器（按干草堆分组建库）、分层抽样
systems/
├── v1_lexical.py            # 规则写入 / BM25 检索 / 计数协议 prompt
├── v2_dense.py              # fastembed bge-small + RRF 混合
├── v3_layered.py            # LLM 原子抽取 / 双层写入 / 冲突失效 / 容量遗忘
├── v4_route.py              # 题型→模型路由 / 假拒答 two-pass 恢复
├── v5_fusion.py             # V3 写入 × V4 读取
└── judge_official.py        # 官方 5 模板 grader prompt（vendored，注明出处）
scripts/ + run.py            # 评测驱动与统一入口
runs/                        # 各版本 JSONL / records / 判分结果
```

## 与 LongMemEval 的关系（诚信声明）

- 数据与判分协议来自官方 [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)：
  `evaluate_qa.py` 的 5 类 grader prompt 逐字引用（`systems/judge_official.py` 顶部注明）；
- 记忆系统本体（写入/存储/检索/更新/遗忘/读取协议）为**独立设计**，未复用任何开源记忆框架代码；
  参考过的思路（RAG 混合检索、生成式智能体记忆流、TimeAware 查询剪枝）在 ITERATIONS.md 中标注了取舍；
- 无任何针对题目的硬编码/特判；检索与读写策略对所有题目统一生效（records 含逐题检索证据，可审计）。

## 文档索引

| 文档 | 内容 |
|---|---|
| [ITERATIONS.md](ITERATIONS.md) | **每次迭代的技术栈选择理由、被否方案、得分与失败模式** |
| [research/](research/) | 立项调研（基准/开源仓库/论文/幻觉抑制/harness 设计） |
| [reports/](reports/) | 评估报告（配置、Overall、分题型、成本、失败模式） |
| runs/ | 评测产物：`hyp_*.jsonl`（交付格式）、逐题 records、判分明细 |

## 交付物对照（手册 1.5）

1. **可运行代码**：本仓库，`run.py` 一键评测 + `lmem` CLI/UI 交互入口
2. **架构设计文档**：ITERATIONS.md（选型理由/取舍）+ 本 README 架构节
3. **Oracle JSONL**：`deliverables/`（500 题 `question_id` + `hypothesis`）
4. **评估报告**：`reports/evaluation_report.md`
5. **答辩材料**：另行准备
