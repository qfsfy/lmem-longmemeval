# -*- coding: utf-8 -*-
"""
mem0 最小可用示例（GLM / 智谱全栈版）

前置条件：
  1) 独立环境已建好（工作区根目录执行过）：
       python -m venv .venv-mem0
       .venv-mem0/Scripts/pip install -e ./reference/mem0
       .venv-mem0/Scripts/pip install fastembed   # 本地 embedding（默认方案，零费用）
  2) 设置智谱 API Key（与你 lmem 项目同一个 ZAI_API_KEY）：
       CMD:        set ZAI_API_KEY=你的Key
       PowerShell: $env:ZAI_API_KEY="你的Key"
       Git Bash:   export ZAI_API_KEY=你的Key
  3) 运行：
       .venv-mem0/Scripts/python scripts/mem0_demo.py

费用模型（2026-10-12 更新）：
  - LLM 走 GLM Coding Plan 专属端点 /api/coding/paas/v4，消耗套餐额度而非账户余额；
    套餐只覆盖对话/代码模型，模型名可用 MEM0_LLM_MODEL 环境变量更换。
  - embedding 默认用本地 fastembed 模型 BAAI/bge-small-zh-v1.5（512 维，零费用零网络，
    首次运行自动下载约 100MB）；Coding Plan 不覆盖 embedding，如坚持用智谱
    embedding-3（需账户余额或资源包），设环境变量 MEM0_EMBEDDER=zhipu。
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("MEM0_TELEMETRY", "False")  # 关闭匿名遥测上报

WORKSPACE = Path(__file__).resolve().parents[1]
STORE_DIR = WORKSPACE / "runs" / "mem0-demo"

CODING_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"  # GLM Coding Plan 专属端点（套餐额度）
OPEN_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"           # 普通按量计费端点（需账户余额）


def build_config():
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        sys.exit(
            "[mem0_demo] 未检测到 ZAI_API_KEY。\n"
            "  先设置密钥再运行，例如 PowerShell:  $env:ZAI_API_KEY=\"你的Key\"\n"
            "  或先用零 Key 版体验全流程:  .venv-mem0/Scripts/python scripts/mem0_demo_offline.py"
        )
    llm_model = os.getenv("MEM0_LLM_MODEL", "glm-5.3-flash")

    # Coding Plan 套餐额度 ≠ 账户余额：对话/代码模型走专属 coding 端点才消耗套餐；
    # embedding 套餐不覆盖，默认本地 fastembed，MEM0_EMBEDDER=zhipu 切回计费版 embedding-3。
    if os.getenv("MEM0_EMBEDDER", "local").lower() == "zhipu":
        embedder = {
            "provider": "openai",
            "config": {"model": "embedding-3", "api_key": api_key,
                       "openai_base_url": OPEN_BASE_URL, "embedding_dims": 1024},
        }
        dims = 1024
    else:
        embedder = {
            "provider": "fastembed",  # 本地 ONNX，首次运行自动下载模型
            "config": {"model": os.getenv("MEM0_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
                       "embedding_dims": 512},
        }
        dims = 512

    return {
        # LLM：mem0 用它做"记忆抽取"（读对话 → 输出 JSON 事实），不是聊天
        "llm": {
            "provider": "openai",  # coding 端点与 OpenAI 协议同构
            "config": {
                "model": llm_model,
                "api_key": api_key,
                "openai_base_url": os.getenv("ZAI_CODING_BASE_URL", CODING_BASE_URL),
                "temperature": 0.1,
            },
        },
        # Embedding：文本 → 向量（默认本地，零费用）
        "embedder": embedder,
        # 向量库：本地 Qdrant（纯文件、无需装服务）
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(STORE_DIR / "qdrant"),
                "collection_name": "mem0_demo",
                "embedding_model_dims": dims,  # 必须与 embedder 一致（mem0 不会自动对齐）
            },
        },
        # 历史账本（SQLite）：记录每条记忆的 ADD/UPDATE/DELETE 与原始消息
        "history_db_path": str(STORE_DIR / "history.db"),
    }


def friendly_exit(e):
    """把智谱端常见的鉴权/计费错误翻译成一句人话。"""
    s, name = str(e), type(e).__name__
    if "401" in s or "Authentication" in name:
        sys.exit("[mem0_demo] 鉴权失败(401)：ZAI_API_KEY 无效，或该 Key 无权访问 coding 端点/该模型。")
    if "429" in s or "RateLimit" in name:
        if "1113" in s:
            sys.exit("[mem0_demo] 429/1113 余额不足：此调用走了按量计费端点。LLM 应消耗 coding "
                     "套餐额度；若你设了 MEM0_EMBEDDER=zhipu，embedding 需要账户余额（open.bigmodel.cn 充值）。")
        sys.exit("[mem0_demo] 429 限流：套餐额度用尽或触发速率限制，稍后重试。")
    raise e


def main():
    from mem0 import Memory

    m = Memory.from_config(build_config())
    print("=" * 70)
    print("mem0 + GLM 最小示例  |  存储:", STORE_DIR)

    # ---------- 1) 写入记忆：LLM 抽取事实 → 向量化 → 存 Qdrant + SQLite 账本 ----------
    print("\n【1】add()：写入两轮对话")
    try:
        r1 = m.add(
        [
            {"role": "user", "content": "我上周和我的狗 Max 去露营了，我们徒步还游泳了。"},
            {"role": "assistant", "content": "听起来很棒！Max 喜欢玩水吗？"},
        ],
        user_id="zhangsan",
    )
    except Exception as e:
        friendly_exit(e)
    for item in r1["results"]:
        print(f"  + 新增记忆: {item['memory']}  (id={item['id'][:8]}…)")

    r2 = m.add(
        [
            {"role": "user", "content": "对了，我下周六约了牙医洗牙，之后要赶高铁去成都出差。"},
            {"role": "assistant", "content": "好的，祝你一切顺利！"},
        ],
        user_id="zhangsan",
    )
    for item in r2["results"]:
        print(f"  + 新增记忆: {item['memory']}  (id={item['id'][:8]}…)")

    # 重复内容 → MD5 指纹去重，应返回空列表
    r3 = m.add("我上周和我的狗 Max 去露营了", user_id="zhangsan")
    print(f"  · 重复写入同一内容 → 抽取结果 {len(r3['results'])} 条（去重生效则为 0）")

    # ---------- 2) 检索记忆：语义 + (可选)BM25 + (可选)实体 加权融合打分 ----------
    for query in ["Max 是谁？", "我最近要去哪里出差？"]:
        print(f"\n【2】search(\"{query}\")")
        hits = m.search(query, filters={"user_id": "zhangsan"}, top_k=3, explain=True)
        if not hits["results"]:
            print("  （无结果）")
        for hit in hits["results"]:
            print(f"  * [{hit['score']:.4f}] {hit['memory']}")
            d = hit.get("score_details")
            if d:
                print(
                    f"      语义={d['semantic_score']:.4f}  BM25={d['bm25_score']:.4f}"
                    f"  实体加分={d['entity_boost']:.4f}"
                    f"  (门槛 threshold={d['threshold']})"
                )

    # ---------- 3) 账本与全量浏览 ----------
    print("\n【3】get_all() 与 history()：")
    all_mems = m.get_all(filters={"user_id": "zhangsan"})["results"]
    print(f"  zhangsan 目前共 {len(all_mems)} 条记忆")
    if all_mems:
        hist = m.history(all_mems[0]["id"])
        print(f"  第一条记忆的操作账本: {len(hist)} 条记录，事件类型 = {[h['event'] for h in hist]}")

    # ---------- 4) 接入你自己的聊天：把检索结果拼进 prompt ----------
    print("\n【4】实际用法示意：检索结果 → 拼进聊天 prompt")
    ctx = "\n".join(f"- {r['memory']}" for r in hits["results"])
    print("  prompt = f'''已知关于用户的记忆：\n{ctx}\n\n用户问题：…'''")

    print("\n完成。数据落盘位置：", STORE_DIR)
    print("（删除 runs/mem0-demo 目录即可彻底重置）")


if __name__ == "__main__":
    main()
