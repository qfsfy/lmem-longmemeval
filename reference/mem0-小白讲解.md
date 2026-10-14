# mem0 项目小白讲解（从零开始读懂一个"AI 记忆层"）

> 阅读对象：第一次接触 LLM 应用开发 / 向量检索的读者。
> 对应源码：`reference/mem0/`（本文所有 `文件:行号` 均指该目录下的代码）。
> 建议配合 `reference/mem0/docs/core-concepts/how-it-works.mdx`（官方"工作原理"，写得最清楚）一起看。

---

## 〇、一句话总结

**mem0（读作 "mem-zero"）是给大模型 AI 助手外挂的一个"长期记忆层"**：你把聊天记录交给它，它用大语言模型（LLM）从中提炼出一条条"事实"，再把每条事实转成一串数字（向量）存进专门的数据库；下次对话前，你把用户的问题交给它，它按"语义相近程度"找出最相关的几条记忆还给你——你把这几十条记忆塞进提示词，AI 就"记住"了这个用户。

一个类比：LLM 是个很聪明但**每次见面都失忆**的专家（它的"大脑"只在一次对话内有效）。mem0 就是给这位专家配的**笔记本 + 图书管理员**：谈话结束后管理员把要点记到卡片上（写入），下次见面前管理员把最相关的几张卡片抽出来放到专家桌上（检索）。

---

## 一、它到底解决什么问题？

1. **LLM 本身没有记忆（无状态）**。每次调用 `chat.completions.create` 都是独立的：这次对话里用户说"我住在成都"，下次再问"我住哪"，模型一无所知——除非你把上一句也发过去。
2. **上下文窗口有限且昂贵**。把全部历史聊天塞进提示词（prompt）里：一是 token 成本随历史长度线性增长；二是历史太长模型反而"看不过来"，重点被稀释（注意力被摊薄），效果下降。
3. **mem0 的答案：不存原文，存"提炼后的记忆"；检索时只取最相关的 top-k 条**。

README 里的成绩单（2026 年 4 月的新算法）：在两个长程记忆基准 LoCoMo 和 LongMemEval 上分别拿到 92.5 和 94.4 分，检索一次只用约 7K token、约 1 秒。注意 README 也明说了：**这些分数是商业平台版（mem0 Platform）跑出来的，开源版方向一致但数字不会完全相同**（平台版有时间推理、图记忆等闭源优化）。

---

## 二、项目全景：目录地图

```
reference/mem0/
├── mem0/                  ← Python 核心库，95% 的逻辑在这
│   ├── memory/
│   │   ├── main.py        ← ★心脏：Memory / AsyncMemory 类（add/search/get_all/update/delete）
│   │   ├── storage.py     ← SQLite 账本：操作历史 + 原始消息
│   │   └── utils.py       ← 消息解析、JSON 提取等杂活
│   ├── embeddings/        ← ★"文本 → 向量"：openai / huggingface / ollama ... 每家一个文件
│   ├── llms/              ← ★"调用大模型"：openai / anthropic / gemini / deepseek ... 每家一个文件
│   ├── vector_stores/     ← ★26 种向量数据库适配：qdrant / chroma / faiss / pgvector ...
│   │   └── base.py        ← 所有向量库的统一接口（抽象基类）
│   ├── configs/
│   │   ├── prompts.py     ← ★所有提示词（LLM 抽取记忆的"工作说明书"）
│   │   └── ...            ← pydantic 配置类（每种组件一个配置）
│   ├── reranker/          ← 重排序器（cohere / huggingface ...）
│   ├── utils/
│   │   ├── scoring.py     ← ★混合检索打分核心（BM25 归一化 + 三信号融合）
│   │   ├── entity_extraction.py ← ★实体抽取（spaCy）
│   │   ├── lemmatization.py     ← 词元还原（attended→attend）
│   │   └── factory.py     ← 工厂：按配置字符串创建组件
│   ├── proxy/             ← OpenAI 兼容的"带记忆聊天"代理
│   └── client/            ← 云平台版客户端 MemoryClient
├── mem0-ts/               ← TypeScript 版本（Node.js 用）
├── server/                ← 自托管服务（FastAPI + Docker + 管理界面）
├── docs/                  ← 文档站源码
├── examples/              ← 各类示例
└── tests/                 ← 测试
```

**三种打开方式**（README 的表格）：本地库（`pip install mem0ai`，适合学习/原型）、自托管 server（`docker compose up`，团队自建）、云平台（注册 app.mem0.ai，零运维）。对学习者，第一种 + 本地向量库（默认 Qdrant/Faiss，无需装服务）就够用。

---

## 三、专业名词详解（按"一条记忆的一生"依次出现）

下面按你在代码里会遇到的顺序讲：**写入时**（LLM、提示词）→ **向量化**（embedding、向量、维度、余弦相似度）→ **存储**（向量数据库、元数据过滤）→ **检索**（top_k、BM25、混合检索、实体、重排序）→ **周边**（去重、账本、异步、工厂）。

### 3.1 LLM（Large Language Model，大语言模型）

GPT、Claude、GLM 这类模型。注意一个容易误解的点：**mem0 用 LLM 不是为了聊天，而是把它当"阅读理解 + 信息整理员"**——输入一段对话，输出结构化的 JSON 事实列表。`llms/` 目录下每个文件是同一件事的不同厂商封装：把消息列表发给 API、拿回文本，仅此而已（见 `mem0/llms/openai.py`，共 150 行）。

### 3.2 Prompt（提示词）与 few-shot（少样本示例）

Prompt 就是发给 LLM 的指令文本。LLM 的行为很大程度上由 prompt 决定，所以 mem0 的"记忆能力"一半藏在 `mem0/configs/prompts.py` 里：

- `FACT_RETRIEVAL_PROMPT`（15 行起，老版）：设定角色"你是个人信息整理师"，要求从对话中抽取事实、以 JSON 输出。里面附了六七个"输入→输出"的示范对，比如输入"Hi, my name is John. I am a software engineer."，输出 `{"facts": ["Name is John", "Is a Software engineer"]}`。**这种在提示词里给例子的手法叫 few-shot（少样本提示）**——不重新训练模型，靠几个例子"教会"它输出格式。
- `ADDITIVE_EXTRACTION_PROMPT`（468 行起，2026 年 4 月新版，当前主流程用的就是它）：核心思想是 **只做 ADD（只增不改不删）**。老版让 LLM 决策"这条事实是新增 ADD、还是更新 UPDATE、还是删除 DELETE"，一次判断错就丢数据；新版让记忆只累积、永不覆盖——"我从奥斯汀搬到西雅图"不会改写"我住奥斯汀"，两条都留着，矛盾留给检索阶段解决。这个设计换来的是：一次 LLM 调用、更省 token、不会因为误判 UPDATE 毁掉记忆。

### 3.3 Embedding（嵌入 / 向量化）

**把一段文字变成一串数字（向量）**，并且保证：意思相近的文字，得到的数字串也"相近"。

```
"我爱吃辣"      → [0.82, -0.15, 0.03, ..., 0.44]   ┐ 方向很接近
"无辣不欢"      → [0.79, -0.11, 0.05, ..., 0.41]   ┘
"明天有雨"      → [-0.3, 0.6, -0.2, ..., 0.1]      ← 方向差很远
```

这件事由专门的 embedding 模型完成（也是调 API）。mem0 的封装在 `mem0/embeddings/openai.py`：默认模型 `text-embedding-3-small`，调用 `embeddings.create()`，返回值 `.data[0].embedding` 就是那串数字（37-55 行）。`embed()` 还有个 `embed_batch()`（57 行）——一次请求算 100 条文本的向量，省网络往返。

### 3.4 向量（Vector）与维度（Dimension）

- **向量**：一个有顺序的数字列表，如 `[0.12, -0.98, 0.45]`。文本被"嵌入"后就成了这样的列表，所以也叫"文本向量"。
- **维度（dimension）**：这个列表的**长度**。`text-embedding-3-small` 输出 **1536 维**，即 1536 个小数（`openai.py:19`）。你可以想象成：模型把每段文字放到一个 1536 维的抽象空间里的某个"位置"，意思越近，位置越近。维数越高通常表达力越强，但存储和计算也更贵。

### 3.5 ★余弦相似度（Cosine Similarity）——本库最核心的度量

**定义：衡量两个向量"方向"有多一致，用夹角 θ 的余弦值表示：**

```
                 A · B  （点积：对应位置相乘再求和）
cos θ = ─────────────────────────────────────
              |A| × |B|  （两个向量各自的"长度/模"）
```

**手算一遍**（三维就够说明问题）：设 A = [1, 0, 1]，B = [1, 0, 0]——

1. 点积 A·B = 1×1 + 0×0 + 1×0 = 1
2. |A| = √(1²+0²+1²) = √2 ≈ 1.414；|B| = √(1²+0²+0²) = 1
3. cos θ = 1 / (1.414 × 1) ≈ **0.707**（对应夹角 45°，方向比较一致）

**怎么解读分数**：结果范围 [-1, 1]。1 = 方向完全相同（意思几乎一样）；0 = 垂直（不相关）；-1 = 相反。embedding 模型通常输出非负语义空间，实际分数大多落在 0~1，**越大越相似**。mem0 里 `search` 默认 `threshold=0.1`（`memory/main.py:1399`），意思是语义相似度低于 0.1 的记忆直接不要。

**为什么用余弦，而不是普通的直线距离（欧氏距离）？** 两个原因：

1. **只关心方向、不关心长短**。"我爱吃辣"和"我特别特别爱吃辣"嵌入后方向几乎一样，但第二个向量可能更长（信息量/词数不同）。欧氏距离会被长度差异干扰，余弦相似度完全不受影响——这正符合"语义相同就算相似"的需求。
2. **embedding 模型在训练时就是按余弦/角度对齐优化的**，用余弦度量是"顺着模型的设计"来用。

**在代码里的落点**：创建向量库集合时指定距离度量，Qdrant 默认 `Distance.COSINE`（`vector_stores/qdrant.py:128`）。注意一个小陷阱：Chroma 库返回的是"余弦**距离**"（= 1 − 余弦相似度，越小越像），所以 mem0 在 `vector_stores/chroma.py:101` 用 `score = 1/(1+距离)` 把它换算回"越大越像"的统一口径。

### 3.6 向量数据库（Vector Store / Vector Database）与 Collection

普通数据库（MySQL）擅长精确查找（"id=5 的行"），不擅长"找出 300 万条向量里和这个查询向量最像的前 20 条"。**向量数据库是专门干这件事的数据库**：

- **Collection（集合）**≈ 关系型数据库里的"表"。mem0 每个用户群的记忆存一个 collection，实体存另一张。
- **ANN（Approximate Nearest Neighbor，近似最近邻）索引**：逐条两两比较是 O(N) 的暴力法，百万条会慢死；向量库内部建特殊索引（如 HNSW 图），牺牲一丁点准确率换取毫秒级返回。你不用配置，库自己干。
- mem0 支持 26 种（`vector_stores/` 目录）：Qdrant、Chroma（轻量本地，适合学习）、Faiss（Meta 的本地库）、Pgvector（PostgreSQL 插件）、云上各种。它们都实现同一个抽象基类 `vector_stores/base.py`（`insert` / `search` / `update` / `delete` / `list`...），所以上层业务代码完全不用改就能换库——这是典型的**依赖倒置/接口抽象**设计。

### 3.7 元数据（Metadata）与过滤（Filters）

每条记忆除了"文本 + 向量"，还挂着一组**元数据**（描述数据的标签数据）：`user_id`（谁的记忆）、`agent_id`（哪个智能体的）、`run_id`（哪次任务运行的）、`created_at`、以及你自定义的任意键值。

**过滤**指：检索时先按这些标签圈定子集（"只看 user=张三 的记忆"），再在这个子集里做相似度搜索。这是**多租户隔离**的关键——没有它，A 用户的体检记录会出现在 B 用户的检索结果里。mem0 还支持高级操作符：`gt/lt/in/contains/AND/OR/NOT` 等（`memory/main.py:1416-1431` 的文档注释），并且对 Qdrant 等远程库自动给 user_id 等字段建过滤索引加速（`qdrant.py:166-184`）。

### 3.8 top_k 与 threshold

- **top_k**：最多返回几条结果。`search` 默认 20；但内部会先取 `max(top_k×4, 60)` 条候选（`memory/main.py:1655`），因为后面还要多信号打分、筛选，先"广撒网"。
- **threshold**：分数门槛（默认 0.1）。语义相似度低于它的候选直接淘汰，防止返回一堆不相关记忆污染提示词。

### 3.9 BM25 与关键词检索、词元还原（Lemmatization）、稀疏向量

余弦相似度擅长"**意思**相近"（"无辣不欢"≈"我爱吃辣"），但对**专有名词**反而不敏感——"Max"和"猫"嵌入后可能距离不远，而检索"Max 得过几岁疫苗"时你需要的是精确含"Max"这个词的那条记忆。所以 mem0 同时跑一路**关键词检索**：

- **BM25**：信息检索领域几十年的经典打分算法（TF-IDF 的改进版）。直观理解：一个词在某条记忆里出现越多、且在全体记忆里越罕见 → 这条记忆对该关键词越相关；同时有参数惩罚"靠堆字数刷分"的长文档。它不走向量，直接对词做匹配。
- **词元还原（lemmatization）**：把词的不同形态归并为"原形"，让 `attended / attends → attend`、`memories → memory` 也能匹配上。实现在 `mem0/utils/lemmatization.py`（用 spaCy 做；注意它刻意保留了 -ing 原形，因为英文 "meeting" 作名词和动词还原结果不同）。写入时每条记忆额外存一份还原后文本 `text_lemmatized`（`memory/main.py:1028`），查询时也把查询还原一遍，两边对齐。
- **稀疏向量（sparse vector）**：BM25 的一种现代存法——一个"几乎全为 0"的超长向量，只有出现过的词的位置有值。Qdrant 等库支持在**同一个集合里同时存稠密向量（语义）和稀疏向量（BM25）**（`qdrant.py:154-163`），一次查询两路齐发。

### 3.10 Sigmoid 归一化与混合检索（Hybrid Search）分数融合

两路检索各给一个分：语义相似度天生在 [0,1]，但 **BM25 原始分没有上界**（可能 3 分也可能 25 分），没法直接相加。mem0 的解法在 `mem0/utils/scoring.py`：

1. **Sigmoid（S 形函数）归一化**：`normalize_bm25 = 1 / (1 + e^(-k·(x−mid)))`（scoring.py:43-54）。直观效果：原始分低于中点 mid → 得分趋近 0；高于 mid → 趋近 1；把任何分数压到 [0,1]。参数还**按查询长度自适应**（16-40 行）：查询词 ≤3 个用 mid=5.0，词越多 mid 越大（长查询天然 BM25 分更高，得把及格线相应上调）。
2. **三信号加权求和**（`score_and_rank`，60-139 行）：

```
总分 = ( 语义分 + BM25分 + 实体加分 ) / 最大可能分
最大可能分 = 1.0 + (有BM25 ? 1.0 : 0) + (有实体 ? 0.5 : 0)   # 动态分母，scoring.py:57,97-101
```

分母动态调整，保证"只用语义"和"三路齐发"时总分口径一致（都是 [0,1]）。`explain=True` 时每条结果还会带 `score_details`，把三个分项拆给你看。**这套"多路召回 + 统一口径 + 加权融合"是记忆系统检索端最值得抄的设计。**

### 3.11 实体抽取（Entity Extraction）、实体库与实体加分

- **实体**：句子里指称具体事物的词——人名、地名、机构、书名、专有概念（"Max"、"旧金山"、"Postgres"）。抽取用 NLP 工具库 **spaCy**（`mem0/utils/entity_extraction.py`）：找大写开头的人名地名、引号里的标题、特定修饰的名词短语（如 "machine learning"）；没装 spaCy 时优雅降级为空。
- **实体库（entity store）**：向量库里单独的一张 collection。写入记忆后（`memory/main.py:1100-1204`，Phase 7），mem0 把这条记忆里的每个实体也向量化存进实体库，并记下 `linked_memory_ids`（这个实体关联哪些记忆）。若实体已存在（文本精确相同，或语义相似度 ≥0.95 视为同一实体），就在它的 `linked_memory_ids` 里追加新记忆 id。
- **实体加分**：检索"Max 是谁？"时（`_search_vector_store` Step 6，`memory/main.py:1675-1678`），从查询里抽出实体 "Max" → 在实体库查到它 → 它关联的所有记忆统一 +0.5 加分（`ENTITY_BOOST_WEIGHT = 0.5`，scoring.py:57）。类比：书的**索引卡**——你查"孙悟空"，索引卡直接告诉你哪些章节提到他，不用一页页翻。

### 3.12 Reranker（重排序）

"先粗筛、再精排"的两段式策略：向量库先便宜地捞出 80 条候选，若 `rerank=True` 且配置了 reranker，就用一个专门的精排模型（Cohere Rerank API、开源的 bge-reranker、甚至让 LLM 来排，见 `mem0/reranker/`）把这 80 条按"与查询的真实相关度"重新排序，取前 top_k。精排模型更准但更贵，所以只对少量候选做。

### 3.13 MD5 哈希去重

写入时对每条新事实的文本算 **MD5 哈希**（一种把任意文本映射成固定长度"指纹"的算法；同样的文本永远得到同样的指纹，`memory/main.py:1022`）。指纹已在库里或本批次出现过 → 判定重复，直接跳过。这是除语义去重外的一道廉价保险。

### 3.14 SQLite 历史账本（history）与原始消息

所有 ADD/UPDATE/DELETE 都会记入一个本地 SQLite 数据库（`mem0/memory/storage.py` 的 `SQLiteManager`），可用 `m.history(memory_id)` 查询单条记忆的变更史。作用：审计（这条记忆是谁、何时写入的）、调试、以及未来做"撤销"。原始对话消息也存在这里（`save_messages`），因为 `add()` 开头要取该用户最近 10 条消息喂给 LLM，用来解决"他/她/那个地方"这类**代词指代**（`memory/main.py:922`）。

### 3.15 其余你会在代码里撞见的词

| 名词 | 一句话解释 | 代码位置 |
|---|---|---|
| UUID | 每条记忆的全局唯一身份证号（128 位随机数） | `main.py:1030` |
| JSON / JSON mode | 结构化数据格式；mem0 强制 LLM 以 JSON 输出（`response_format={"type":"json_object"}`），解析失败还有 `extract_json` 兜底 | `main.py:963,982` |
| 同步/异步（AsyncMemory） | 同一套逻辑两份实现；异步版 await 期间不阻塞线程，Web 高并发场景必需 | `main.py:2193` |
| 工厂模式（Factory） | 按 `config.llm.provider="openai"` 这类字符串自动 new 出对应类，换厂商只改配置不改代码 | `mem0/utils/factory.py` |
| pydantic | Python 的"配置即类"库，自动做类型校验，mem0 所有配置类基于它 | `configs/base.py` |
| Telemetry（遥测） | 匿名上报"你用了哪个向量库"等统计，帮助官方改进；不影响功能 | `memory/telemetry.py` |
| RAG | 检索增强生成——先检索资料再让 LLM 作答的通用范式，mem0 本质是面向"用户记忆"的专用 RAG 基础设施 | — |

---

## 四、跟着一条记忆走完全程（两条核心流水线）

### 4.1 写入：`add()` 的 8 个 Phase（`memory/main.py:881-1220`）

场景：用户对 AI 说——"我上周和我的狗 Max 去露营了，我们徒步还游泳了。"

| Phase | 做什么 | 代码 |
|---|---|---|
| 0 上下文收集 | 从 SQLite 取该用户最近 10 条消息（让 LLM 能理解"我的狗"指谁） | 920-923 |
| 1 检索旧记忆 | 把新消息向量化 → 在该用户的记忆里搜最相关的 10 条；**把记忆的 UUID 映射成 0,1,2… 序号再给 LLM**（防止 LLM 编造 UUID，注释原话 anti-hallucination，938-940 行——很值得学的细节） | 925-940 |
| 2 LLM 抽取 | 一次调用：系统提示词 = ADDITIVE_EXTRACTION_PROMPT，输出 `{"memory":[{"text":"User went camping with their dog Max...", ...}]}` | 942-986 |
| 3 批量向量化 | `embed_batch` 一次算出所有新事实的向量 | 993-1005 |
| 4/5 去重 | MD5 指纹与"库内已有 + 本批次内"双重比对，重复即弃 | 1007-1026 |
| 6 批量写入 | 向量 + 元数据一次插入向量库（失败降级为逐条）；写入成功才返回成功 | 1047-1076 |
| 记账 | 批量写入 SQLite 历史账本 | 1078-1098 |
| 7 实体链接 | spaCy 抽实体 → 批量向量化 → 实体库中找同类（精确/语义≥0.95）→ 挂接或新建 | 1100-1204 |
| 8 收尾 | 保存原始消息；返回 `[{"id","memory","event":"ADD"}]` | 1206-1220 |

注意 `add(messages, infer=False)` 的旁路：跳过 LLM 抽取，原文直接当记忆存（README 也说了适合"必须原样保存"的场景）。

### 4.2 检索：`search()` 的 9 步打分（`_search_vector_store`，`memory/main.py:1642-1745`）

场景：三天后用户问——"Max 是谁？"

1. 查询预处理：词元还原（给 BM25 用）+ 实体抽取（得到 `["Max"]`）；
2. 查询向量化；
3. **语义搜索**：取 80 条候选（`max(top_k×4, 60)`）；
4. **关键词搜索**：BM25 又取 80 条；
5. BM25 原始分 → 按查询长度自适应的 sigmoid 归一化；
6. **实体加分**：实体库查 "Max" → 其 `linked_memory_ids` 里的记忆 +0.5；
7. 组装候选集（顺带过滤已过期的记忆 `expiration_date`）；
8. `score_and_rank`：先 threshold 卡语义分 → 三信号相加除以动态最大分 → 排序取 top 20；
9. 格式化输出 `{"results":[{"id","memory","score":0.87,"created_at",...}]}`，`explain=True` 时附上每个分项。

---

## 五、跑通最小示例（本工作区已验证的方案）

> 实际操作直接用工作区里的三个现成脚本（2026-10-12 配置更新为 coding 端点 + 本地 embedding）：
>
> | 脚本 | 用途 | 前置条件 |
> |---|---|---|
> | `scripts/mem0_demo_offline.py` | 零 Key 全链路体验（mock 掉 OpenAI 传输层） | 无 |
> | `scripts/mem0_demo.py` | 真实 GLM 全栈（coding 端点抽取 + 本地 fastembed 向量化） | `ZAI_API_KEY` |
> | `scripts/mem0_agent_chat.py` | 带记忆的交互聊天（两阶段读写样板；加 `--offline` 可离线） | `ZAI_API_KEY` |
>
> ```bash
> # 一次性环境准备（已完成）：独立 venv + 从本地仓库安装 2.2.1
> python -m venv .venv-mem0
> .venv-mem0/Scripts/pip install -e ./reference/mem0
>
> # 零 Key 先跑通管线
> .venv-mem0/Scripts/python scripts/mem0_demo_offline.py
>
> # 真实 GLM 版（和你 lmem 项目同一个 Key）
> export ZAI_API_KEY=你的Key        # PowerShell: $env:ZAI_API_KEY="你的Key"
> .venv-mem0/Scripts/python scripts/mem0_demo.py
> ```
>
> 三个新手必知的坑（都写进了脚本注释）：
> 1. **别用 pip 全局装的 mem0ai**：PyPI 上是 2.1.0，与仓库 2.2.1 代码结构不同，跑的代码要和你读的代码一致；
> 2. **维度必须手动对齐**：`embedder.config.embedding_dims` 必须等于 `vector_store.config.embedding_model_dims`，mem0 不会自动同步（见 `vector_stores/qdrant.py:12`，默认都是 1536）；
> 3. **智谱 embedding-3 支持 dimensions 参数**（256~2048，默认 2048），走 OpenAI 兼容协议直连 `https://open.bigmodel.cn/api/paas/v4`。
> 4. **Coding Plan 套餐额度 ≠ 账户余额**：GLM Coding Plan 走专属端点 `https://open.bigmodel.cn/api/coding/paas/v4`，只覆盖对话/代码模型（glm-5.3-flash 等）；embedding 等其他模型只能走 `paas/v4` 按量计费，套餐不报销。所以脚本默认 LLM 走 coding 端点、embedding 用本地 fastembed（`BAAI/bge-small-zh-v1.5`，512 维，`pip install fastembed`）；设 `MEM0_EMBEDDER=zhipu` 才切回计费版 embedding-3。
>
> 可选增强：`pip install fastembed` 解锁 BM25 关键词检索；`pip install "mem0ai[nlp]"` + `python -m spacy download en_core_web_sm` 解锁实体加分（两者缺失均自动降级，只影响打分不影响运行）。

mem0 的 LLM/embedding 封装都走 **OpenAI 兼容协议**，核心配置就三块（摘自 `scripts/mem0_demo.py`）：

```python
from mem0 import Memory

config = {
    "llm": {  # 做记忆抽取：走 Coding Plan 专属端点，消耗套餐额度
        "provider": "openai",
        "config": {"model": "glm-5.3-flash", "api_key": KEY,
                   "openai_base_url": "https://open.bigmodel.cn/api/coding/paas/v4"},
    },
    "embedder": {  # 文本 → 向量：默认本地 fastembed，零费用零网络
        "provider": "fastembed",
        "config": {"model": "BAAI/bge-small-zh-v1.5", "embedding_dims": 512},
    },
    "vector_store": {  # 本地 Qdrant，纯文件无需装服务
        "provider": "qdrant",
        "config": {"path": "runs/mem0-demo/qdrant", "collection_name": "mem0_demo",
                   "embedding_model_dims": 512},   # 与 embedder 一致！
    },
    "history_db_path": "runs/mem0-demo/history.db",
}
m = Memory.from_config(config)

m.add("我上周和我的狗 Max 去露营了", user_id="zhangsan")            # 写入
hits = m.search("Max 是谁？", filters={"user_id": "zhangsan"}, top_k=5)  # 检索
context = "\n".join(f"- {r['memory']}" for r in hits["results"])    # 拼进聊天 prompt
```

---

## 六、对你自己项目（lmem harness）的可借鉴点

结合你 README 里"已有 SQLite + 进程内 BM25、双时间轴记忆卡、抽取→冲突检测→失效不删除"的现状：

1. **三信号融合打分直接可抄**：`utils/scoring.py` 总共 139 行，语义 + BM25 + 实体的"动态分母归一化"比拍脑袋定权重优雅得多；等你接入稠密检索后，这套就是现成的融合层。
2. **实体库做"直取通道"**：你 retriever 里的"画像直取"可以升级成 mem0 这种 实体→linked_memory_ids 的倒排结构，专名查询命中更稳。
3. **ADD-only 与"失效不删除"互相印证**：你的双时间轴（valid_from/to + observed_at）和 mem0 v3 的"只增不改、检索时解决矛盾"是同一哲学的两个实现，mem0 用基准分数证明了这条路的有效性。
4. **UUID→序号的防幻觉技巧**：任何"把数据库 id 交给 LLM 去引用"的场景都适用。
5. **批量 embed / 批量 insert + 失败降级**：成本与时延的工程细节，`add()` 的 Phase 3-6 是很好的模板。
6. **警惕 OSS 与平台版的差距**：README 明说时间推理（temporal）在开源版会直接抛错提示"平台专属"（`main.py:817-818,1446-1447`）。你评测 LongMemEval 时的"时间感知"必须自研，mem0 开源版给不了参考实现。

---

## 七、推荐的源码阅读路线（新人版）

1. `docs/core-concepts/how-it-works.mdx` —— 15 分钟建立心智模型（本文第四节即它的代码版）；
2. 跑通第五节的最小示例，用默认本地 Qdrant，观察 `results` 的结构；
3. `mem0/memory/main.py:487` `Memory.__init__` —— 看四大组件（embedder / vector_store / llm / db）如何被工厂创建、组装；
4. `add()` 八个 Phase（881-1220 行），对照本文 4.1 逐段读；
5. `search()` 九步（1642-1745 行）+ `utils/scoring.py` 全文（139 行，务必逐行）；
6. `configs/prompts.py` 的 `ADDITIVE_EXTRACTION_PROMPT`（468 行起）——记忆质量的源头；
7. 挑一个向量库实现读：建议 `vector_stores/qdrant.py`（功能最全：稠密+BM25 稀疏+过滤索引），对照 `vector_stores/base.py` 的接口看"适配器"如何落地；
8. 最后对照 `AsyncMemory`（2193 行起）看异步版差异——逻辑相同，只是换 `await` 写法。
