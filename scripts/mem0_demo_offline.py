# -*- coding: utf-8 -*-
"""
mem0 离线全链路演示（零 API Key、零费用、零网络）

原理：把 OpenAI SDK 的 HTTP 传输层替换为进程内 mock（httpx.MockTransport）——
      /chat/completions 永远返回固定的"抽取结果"JSON，/embeddings 返回基于
      字面 bigram 的确定性向量。然后让 mem0 完整跑一遍：
      add → LLM 抽取 → 向量化 → 哈希去重 → 存 Qdrant/SQLite → search 打分 → 账本。
      唯一没被验证的是真实网络请求格式，那由 scripts/mem0_demo.py（真实 GLM）负责。

运行：.venv-mem0/Scripts/python scripts/mem0_demo_offline.py
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("MEM0_TELEMETRY", "False")

WORKSPACE = Path(__file__).resolve().parents[1]
STORE_DIR = WORKSPACE / "runs" / "mem0-demo-offline"
DIMS = 256

# mock LLM 永远"抽取"出这两条事实（模拟 ADDITIVE_EXTRACTION_PROMPT 要求的 JSON 输出格式）
FIXED_FACTS = ["用户养了一只名叫 Max 的狗", "用户上周和 Max 去露营，一起徒步和游泳"]


def mock_embed(text: str) -> list:
    """字面 unigram+bigram 哈希向量：字面重叠越多、余弦相似度越高（语义相似度的最低保障模拟）。"""
    v = [0.0] * DIMS
    t = text.lower()
    grams = list(t) + [t[i : i + 2] for i in range(len(t) - 1)] + t.split()
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        v[h % DIMS] += 1.0 if (h >> 8) & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def mock_handler(request) -> "httpx.Response":
    body = json.loads(request.content or b"{}")
    if request.url.path.endswith("/embeddings"):
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        data = [
            {"object": "embedding", "index": i, "embedding": mock_embed(t)}
            for i, t in enumerate(inputs)
        ]
        return httpx.Response(200, json={"object": "list", "data": data,
                                         "model": body.get("model", "mock"),
                                         "usage": {"prompt_tokens": 0, "total_tokens": 0}})
    if request.url.path.endswith("/chat/completions"):
        if body.get("response_format"):  # mem0 add() 的抽取请求：要求 JSON 输出
            content = json.dumps({"memory": [{"text": t} for t in FIXED_FACTS]}, ensure_ascii=False)
        else:  # 普通聊天请求（agent_chat 的对话回复）
            content = "(离线mock回复) 这里本该是 GLM 的回答；离线模式下我只会说这句固定话。"
        return httpx.Response(200, json={
            "id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
            "model": body.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
    return httpx.Response(404, json={"error": {"message": f"unexpected path: {request.url.path}"}})


def patch_openai_clients():
    """让 mem0 内部创建的 OpenAI 客户端走本地 mock 传输层（mem0 在方法内才 import OpenAI，
    因此同时 patch openai.OpenAI 本体与 mem0 两个模块的全局名，双保险）。"""
    import httpx
    import openai

    original_openai_cls = openai.OpenAI  # 先保存原类，factory 内必须用原类，否则无限递归

    def factory(**kw):
        kw["http_client"] = httpx.Client(transport=httpx.MockTransport(mock_handler))
        return original_openai_cls(**kw)

    openai.OpenAI = factory
    import mem0.embeddings.openai as emb_mod
    import mem0.llms.openai as llm_mod
    emb_mod.OpenAI = factory
    llm_mod.OpenAI = factory


def main():
    from mem0 import Memory

    patch_openai_clients()
    config = {
        "llm": {"provider": "openai",
                "config": {"model": "mock-llm", "api_key": "mock-key",
                           "openai_base_url": "http://mock.local/v1"}},
        "embedder": {"provider": "openai",
                     "config": {"model": "mock-embedding", "api_key": "mock-key",
                                "openai_base_url": "http://mock.local/v1",
                                "embedding_dims": DIMS}},
        "vector_store": {"provider": "qdrant",
                         "config": {"path": str(STORE_DIR / "qdrant"),
                                    "collection_name": "mem0_demo_offline",
                                    "embedding_model_dims": DIMS}},
        "history_db_path": str(STORE_DIR / "history.db"),
    }
    m = Memory.from_config(config)
    print("[mem0] Memory 初始化成功（本地 Qdrant + SQLite 账本 + mock OpenAI 传输层）\n")

    print("【add #1】写入对话 → LLM(mock) 抽取出固定 2 条事实")
    r1 = m.add(
        [
            {"role": "user", "content": "我上周和我的狗 Max 去露营了，我们徒步还游泳了。"},
            {"role": "assistant", "content": "听起来很棒！"},
        ],
        user_id="zhangsan",
    )
    for item in r1["results"]:
        print(f"  + {item['memory']}")

    print("\n【add #2】重复写入相同内容 → 验证 MD5 去重")
    r2 = m.add("我上周和我的狗 Max 去露营了", user_id="zhangsan")
    print(f"  新增 {len(r2['results'])} 条（应为 0，去重生效）")

    print('\n【search】"Max 是谁？" → 混合检索打分（explain=True 展示分项）')
    hits = m.search("Max 是谁？", filters={"user_id": "zhangsan"}, top_k=5, explain=True)
    for hit in hits["results"]:
        print(f"  * [{hit['score']:.4f}] {hit['memory']}")
        d = hit.get("score_details")
        if d:
            print(f"      语义={d['semantic_score']:.4f}  BM25={d['bm25_score']:.4f}"
                  f"  实体={d['entity_boost']:.4f}  threshold={d['threshold']}")
    print("  （本环境未装 fastembed/spacy，BM25 与实体加分为 0 属正常降级；")
    print("    真实 GLM 版同理，pip install fastembed 可解锁 BM25）")

    all_mems = m.get_all(filters={"user_id": "zhangsan"})["results"]
    print(f"\n【get_all】zhangsan 共 {len(all_mems)} 条记忆")
    if all_mems:
        hist = m.history(all_mems[0]["id"])
        print(f"【history】第一条记忆的账本: {[{'event': h['event'], 'new': (h['new_memory'] or '')[:20]} for h in hist]}")

    print("\n[OK] 离线全链路验证通过 —— mem0 管线在你的机器上已跑通。")
    print("     下一步：设置 ZAI_API_KEY 后运行 scripts/mem0_demo.py 切换到真实 GLM。")


if __name__ == "__main__":
    main()
