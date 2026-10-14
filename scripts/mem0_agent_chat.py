# -*- coding: utf-8 -*-
"""
带记忆的最小 Agent 聊天循环 —— mem0 官方推荐的"两阶段"集成模式标准样板

    ┌─────────────────────── 每一轮对话 ───────────────────────┐
    │  用户输入                                                 │
    │    ↓ ①读: search(用户输入) → 取最相关记忆                  │
    │  记忆拼进 system prompt → 调 LLM 生成回复                  │
    │    ↓ ②写: add(本轮对话) → LLM 抽取事实入库                 │
    └──────────────────────────────────────────────────────────┘

记忆不是 mem0 自己"偷听"来的，全部来自你代码里这两个调用点。

用法：
  真实:  .venv-mem0/Scripts/python scripts/mem0_agent_chat.py     （需 ZAI_API_KEY）
  离线:  .venv-mem0/Scripts/python scripts/mem0_agent_chat.py --offline
退出：输入 exit / quit / q / 退出
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("MEM0_TELEMETRY", "False")

WORKSPACE = Path(__file__).resolve().parents[1]
STORE_DIR = WORKSPACE / "runs" / "mem0-agent-chat"
DIMS = 256
USER_ID = "demo_user"


def build_real_config():
    """真实 GLM 配置（与 mem0_demo.py 相同的三件套）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mem0_demo import build_config
    return build_config()


def build_offline_config():
    """零 Key 离线配置：patch 掉 OpenAI 传输层 + 本地 Qdrant。"""
    from mem0_demo_offline import patch_openai_clients
    patch_openai_clients()
    return {
        "llm": {"provider": "openai",
                "config": {"model": "mock-llm", "api_key": "mock-key",
                           "openai_base_url": "http://mock.local/v1"}},
        "embedder": {"provider": "openai",
                     "config": {"model": "mock-embedding", "api_key": "mock-key",
                                "openai_base_url": "http://mock.local/v1",
                                "embedding_dims": DIMS}},
        "vector_store": {"provider": "qdrant",
                         "config": {"path": str(STORE_DIR / "qdrant"),
                                    "collection_name": "mem0_agent_chat",
                                    "embedding_model_dims": DIMS}},
        "history_db_path": str(STORE_DIR / "history.db"),
    }


SYSTEM_TEMPLATE = """你是用户的长期 AI 助手。以下是历史对话沉淀的关于该用户的记忆，
可能不完整或过时：回答时优先采信与问题最相关、created_at 最新的记忆。

{context}

要求：
- 记忆与问题无关时不要硬扯；
- 涉及用户个人事实但记忆里没有依据时，坦白说"我不确定"，不要编造。"""


def main():
    offline = "--offline" in sys.argv
    from mem0 import Memory

    config = build_offline_config() if offline else build_real_config()
    m = Memory.from_config(config)
    if not offline and not os.getenv("ZAI_API_KEY"):
        sys.exit("[agent_chat] 未检测到 ZAI_API_KEY。真实版需先设置；或加 --offline 跑离线版。")

    if offline:
        print("[离线模式] LLM 回复与抽取均为固定 mock，仅演示数据流。")
    print(f"用户标识 user_id = {USER_ID}；输入 exit 退出。\n")

    while True:
        try:
            user_msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_msg or user_msg.lower() in {"exit", "quit", "q", "退出"}:
            break

        # ---------- ① 读：检索记忆（生产中这一步在"回答前"发生） ----------
        hits = m.search(user_msg, filters={"user_id": USER_ID}, top_k=5)["results"]
        context = "\n".join(f"- [{h['score']:.2f}] {h['memory']}" for h in hits) or "（暂无记忆）"

        # ---------- 生成：记忆 + 问题 → LLM ----------
        reply = m.llm.generate_response(messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)},
            {"role": "user", "content": user_msg},
        ])
        print(f"助手: {reply}")
        print(f"  [读] 本轮注入 {len(hits)} 条记忆:")
        for h in hits:
            print(f"     · ({h['score']:.2f}) {h['memory']}")

        # ---------- ② 写：本轮对话沉淀为记忆（生产中应放后台线程，别阻塞用户） ----------
        added = m.add(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": reply}],
            user_id=USER_ID,
        )
        print(f"  [写] 本轮新增 {len(added['results'])} 条记忆")
        for item in added["results"]:
            print(f"     + {item['memory']}")

    total = m.get_all(filters={"user_id": USER_ID})["results"]
    print(f"\n会话结束。{USER_ID} 的记忆库共 {len(total)} 条（已持久化到 {STORE_DIR}）")
    print("下次再运行本脚本，AI 依然记得这些内容 —— 这就是长期记忆。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        s, name = str(e), type(e).__name__
        if "401" in s or "Authentication" in name:
            sys.exit("\n[agent_chat] 鉴权失败(401)：ZAI_API_KEY 无效，或该 Key 无权访问 coding 端点/该模型。")
        if "429" in s or "RateLimit" in name:
            if "1113" in s:
                sys.exit("\n[agent_chat] 429/1113 余额不足：此调用走了按量计费端点。LLM 应消耗 "
                         "coding 套餐额度；若设了 MEM0_EMBEDDER=zhipu，embedding 需要账户余额。")
            sys.exit("\n[agent_chat] 429 限流：套餐额度用尽或触发速率限制，稍后重试。")
        raise
