# 长交互式AI记忆系统（Long-term Interactive AI Memory System）

> 立项日期：2026-09-21 ｜ 当前阶段：技术调研与积累

## 项目定位

构建面向长程多轮交互的 AI 记忆与上下文管理系统，解决：

1. **复杂记忆管理**——海量历史会话中的信息抽取、整合、更新、遗忘与检索；
2. **幻觉抑制**——通过接地（grounding）、知识更新失效、校准拒答等机制减少编造。

**最终产物形态**：一个可复用的记忆 **harness**（写入管线 + 存储检索 + 上下文装配 + 生成门控），主要搭配 **GLM-5.3-Flash**（快/省档，抽取与分析主力）与 **GLM-5.3**（旗舰档，复杂推理与仲裁）双模型运行。**评测目标为 LongMemEval Oracle 档（500 题，`longmemeval_oracle.json`）**，属 D.I.V.E.（赛诺菲×川大）个人赛交付：代码 + 架构文档 + Oracle JSONL + 评估报告 + 答辩（评分 40/20/15/15/10，详见 research/07）。

## 快速开始（harness 已可运行）

```bash
# 1) 安装（Python ≥3.10）
pip install -e ".[ui,dev]"

# 2a) 无 API Key 也能玩：自动进入演示模式（Mock 后端，离线规则）
lmem ui                # Web 界面 → 点"一键载入示例记忆" → 问"我现在住在哪里？"

# 2b) 接入真实 GLM：设置 Key 后重启即切换
set ZAI_API_KEY=...    # 或写入 .env；端点见 configs/default.yaml
lmem probe             # 实测连通性/时延/（可选）长上下文
```

更多命令：`lmem ask / chat / ingest / probe / eval run`（`lmem --help`）。
冒烟测试：`python -m pytest tests -q`（8 项，全离线）。

### 代码结构

```
src/lmem/
├── config.py            # 配置与默认值（configs/default.yaml 可覆盖）
├── llm/client.py        # OpenAI 兼容后端 + Mock 后端（任务级：extract/analyze/generate/judge）
├── memory/
│   ├── schema.py        # 记忆卡（双时间轴 valid_from/to + observed_at，失效不删除）
│   ├── store.py         # SQLite 持久化 + 进程内 BM25（中文单字+二字组分词）
│   ├── writer.py        # 写入管线：抽取 → 冲突检测 → 失效/版本化
│   └── retriever.py     # 查询分析 → BM25+画像直取 → 证据打包+覆盖率
├── pipeline/
│   ├── router.py        # flash/flagship 路由（cascade / by_type / …）
│   ├── generator.py     # 结构化生成 {answer, citations, confidence, abstained}
│   ├── gate.py          # 拒答门控（反幻觉出口）
│   └── harness.py       # MemoryHarness 门面
├── evals/               # LongMemEval 适配器 / 快速 judge / 分题型指标
├── ui/app.py            # Gradio 界面（对话 / 记忆库 / 设置）
├── probe.py             # 模型端点探测
└── cli.py               # lmem 命令行
```

当前为**脚手架阶段**：稠密检索、图联想、LLM 覆盖率评估、时间感知查询改写等按
`research/05`、`research/06` 的路线分阶段接入（代码内已留 `roadmap` 注释锚点）。

## 最终评测目标

**LongMemEval**（ICLR 2025）——500 题的长程交互记忆基准，即最初记忆中"那个 500 题的题库"。
详见 `research/01-目标基准LongMemEval.md`。

## 文档索引

| 文档 | 内容 |
|---|---|
| [research/01-目标基准LongMemEval.md](research/01-目标基准LongMemEval.md) | 500 题基准的构成、题型、评测协议、相关基准对比 |
| [research/02-开源记忆系统仓库.md](research/02-开源记忆系统仓库.md) | Mem0 / Letta / Zep / MemOS / A-MEM 等开源仓库分类清单 |
| [research/03-论文阅读清单.md](research/03-论文阅读清单.md) | 按优先级分组的论文清单（P0/P1/P2） |
| [research/04-幻觉抑制专题.md](research/04-幻觉抑制专题.md) | 幻觉成因、三条抑制路线、与五类题型的机制映射 |
| [research/05-技术路线与下一步.md](research/05-技术路线与下一步.md) | 分阶段研发路线图、工程栈选型、风险清单 |
| [research/06-Harness设计-GLM双模型.md](research/06-Harness设计-GLM双模型.md) | **最终产物设计**：记忆 harness 七模块架构 + GLM-5.3/5.3-Flash 双模型分工 |
| [research/07-学生手册要点与执行计划.md](research/07-学生手册要点与执行计划.md) | **D.I.V.E. 手册落地**：Oracle 档评测、评分构成、差距清单、四阶段计划 |

## 当前状态与下一步

- [x] 确认目标基准为 LongMemEval（500 题）
- [x] 完成首轮开源仓库与论文普查
- [x] **harness 脚手架搭建完成**（Mock 全流程可跑、CLI + Web UI、8 项冒烟测试通过）
- [x] **学生手册落地**：评测确认为 Oracle 档、评分/交付物/红线对齐（research/07）
- [ ] 本周：git init + GitHub 建仓（真实 commit 历史是评分项）；配 `ZAI_API_KEY` 跑 `lmem probe`
- [ ] Phase 1（→11 月上旬）：下载 `longmemeval_oracle.json`，30 题小规模验证 + 成本估算，**提交个人技术方案**
- [ ] Phase 2（10/10–11/30）：LLM 抽取替换规则抽取；补齐必做缺口（遗忘/生命周期策略、episodic 原文层、效率打点）
- [ ] Phase 3（11 月下旬–12 月上旬）：全量 500 题评测 + 分题型失败模式分析 + 评估报告
- [ ] Phase 4（12 月）：文档收口、Code freeze、答辩演练

