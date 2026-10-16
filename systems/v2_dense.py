"""V2「稠密语义」记忆系统 —— V1 的规则写入不变，检索升级为 BM25 + 向量余弦 RRF 混合。

动机（来自 V1 失败模式分析）：
- BM25 是词面匹配，"battery life tips" 召不回 "portable power bank"、"embroidery 体验" 召不回
  情感归属词 —— 这类词汇鸿沟失败在 single-session-assistant / preference 题型集中出现；
- 稠密向量弥补语义泛化，BM25 保住专有名词/数字的精确匹配，RRF 融合两者排序。

工程选型：
- fastembed(ONNX bge-small-en-v1.5, 384 维)：CPU 即可、无 torch 依赖（装包体积 2.5GB+ 的坑）、
  自带 query/passage 前缀处理；模型文件缓存在 G 盘（C 盘满）；
- 向量计算 numpy 暴力余弦：单用户库 ≤ 数千条，毫秒级，无需 ANN 索引；
- 向量存 SQLite BLOB 列（同库同事务），不引第三方向量库服务。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lmem.config import ModelConfig
from lmem.evals.longmemeval import group_by_haystack
from lmem.llm.client import OpenAICompatBackend
from lmem.memory.schema import MemoryCard, Session
from lmem.memory.store import MemoryStore

from systems.v1_lexical import (
    QTYPE_HINTS,
    GEN_SYSTEM,
    analyze_query_rules,
    deepseek_config,
    ingest_rules,
    load_env,
    stratified_subset,
)

# --------------------------------------------------------------------------
# 嵌入索引（进程内，随组生命周期）
# --------------------------------------------------------------------------

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding
        _MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _MODEL


def _l2(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(n, 1e-9)


class DenseIndex:
    """组内稠密索引：card_id -> 归一化向量；embedder 可插拔（智谱 API / 本地 fastembed）。"""

    def __init__(self, embedder=None) -> None:
        if embedder is None:
            from systems.embedders import make_embedder
            embedder = make_embedder()
        self.embedder = embedder
        self.ids: list[str] = []
        self.mat: np.ndarray | None = None

    def add(self, cards: list[MemoryCard]) -> None:
        if not cards:
            return
        texts = [c.content for c in cards]
        vecs = self.embedder.embed(texts)
        mat = _l2(np.array(vecs, dtype=np.float32))
        self.ids.extend(c.id for c in cards)
        self.mat = mat if self.mat is None else np.vstack([self.mat, mat])

    def search(self, question: str, allowed: set[str], k: int) -> list[tuple[str, float]]:
        if self.mat is None or not self.ids:
            return []
        q = _l2(np.array(list(self.embedder.query_embed([question])), dtype=np.float32))[0]
        sims = self.mat @ q
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for idx in order:
            cid = self.ids[idx]
            if cid in allowed:
                out.append((cid, float(sims[idx])))
            if len(out) >= k:
                break
        return out


def retrieve_hybrid(store: MemoryStore, index: DenseIndex, question: str,
                    as_of: str | None, bm25_top_k: int = 24, dense_top_k: int = 24,
                    rrf_k: int = 60):
    info = analyze_query_rules(question)
    date = None
    if as_of and not info["include_past"] and info["qtype"] != "knowledge_update":
        date = as_of
    pool = store.all_cards() if date is None else store.active_at(date)
    allowed = {c.id for c in pool}

    probe = store.search(question, k=1, date=date)
    bm25_top = probe[0][1] if probe else 0.0

    # 记忆召回预算：聚合题两路各放宽到 48，仍按相关性排序（不拼塞全库）
    k = 48 if info["qtype"] == "multi_session" else bm25_top_k
    bm25_hits = store.search(question, k=k, date=date)
    dense_hits = index.search(question, allowed, k)

    # Reciprocal Rank Fusion
    scores: dict[str, float] = {}
    for rank, (card, _) in enumerate(bm25_hits):
        cid = card.id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (cid, sim) in enumerate(dense_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    by_id = {c.id: c for c in pool}
    cards = [by_id[cid] for cid, _ in ranked if cid in by_id]
    cos_top = dense_hits[0][1] if dense_hits else 0.0

    existing = {c.id for c in cards}
    profiles = [c for c in pool if c.kind == "profile"]
    profiles.sort(key=lambda c: c.observed_at, reverse=True)
    cards += [c for c in profiles[:5] if c.id not in existing]
    return cards, bm25_top, info, cos_top


# --------------------------------------------------------------------------
# 评测流水线（复用 V1 生成协议）
# --------------------------------------------------------------------------

def answer_question(store: MemoryStore, index: DenseIndex, backend: OpenAICompatBackend,
                    model_cfg: ModelConfig, question: str, as_of: str | None,
                    abstain: tuple[float, float] = (0.0, 0.0)) -> tuple[str, dict]:
    cards, bm25_top, info, cos_top = retrieve_hybrid(store, index, question, as_of)
    meta = {"qtype": info["qtype"], "top_bm25": round(bm25_top, 2),
            "top_cos": round(cos_top, 3), "n_evidence": len(cards),
            "evidence_chars": sum(len(c.content) for c in cards)}
    bm25_min, cos_min = abstain
    if not cards or (bm25_top < bm25_min and cos_top < cos_min):
        meta["abstain"] = "score_gate"
        return "I don't have that information in my memory.", meta
    hint = QTYPE_HINTS.get(info["qtype"], "")
    lines = [f"[{i}] ({c.valid_from}{'→' + c.valid_to if c.valid_to else ''}) {c.content}"
             for i, c in enumerate(cards, 1)]
    user = (f"Question date: {as_of or 'unknown'}\nQuestion: {question}\n"
            + (f"Hint: {hint}\n" if hint else "")
            + "\nMemories:\n" + ("\n".join(lines) if lines else "(none)"))
    comp = backend.complete(task="generate", model_cfg=model_cfg, system=GEN_SYSTEM,
                            user=user, extra={"question": question})
    ans = (comp.text or "").strip() or "I don't have that information in my memory."
    meta["usage"] = comp.usage
    return ans, meta


def run(questions: list[dict], out_dir: str | Path, tag: str = "v2",
        model_cfg: ModelConfig | None = None,
        abstain: tuple[float, float] = (0.0, 0.0), limit: int | None = None,
        workers: int = 3) -> dict:
    from systems.runner import run_pipeline
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = model_cfg or deepseek_config()
    backend = OpenAICompatBackend()

    if limit:
        questions = questions[:limit]
    groups = group_by_haystack(questions)

    def build_group(g):
        store = MemoryStore(":memory:")
        for s in g["sessions"]:
            ingest_rules(store, s)
        index = DenseIndex()
        index.add(store.all_cards())
        return store, index

    def answer_one(ctx, q):
        store, index = ctx
        try:
            ans, meta = answer_question(store, index, backend, model_cfg,
                                        q["question"], q.get("question_date"), abstain)
        except Exception as e:
            ans = "I don't have that information in my memory."
            meta = {"qtype": "error", "error": str(e)[:200]}
        return ans, meta

    t0 = time.time()
    records, usage_total = run_pipeline(
        groups, build_group, answer_one, out_dir, tag, workers=workers)

    summary = {"tag": tag, "n_questions": len(records),
               "elapsed_s": round(time.time() - t0, 1), "usage": usage_total,
               "qtype_dist": dict(Counter(r["question_type"] for r in records))}
    (out_dir / f"run_summary_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/longmemeval/longmemeval_oracle.json")
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bm25-min", type=float, default=0.0)
    ap.add_argument("--cos-min", type=float, default=0.0)
    args = ap.parse_args()
    load_env()
    questions = json.loads(Path(args.data).read_text(encoding="utf-8"))
    tag = "v2_sub%d" % args.subset if args.subset else "v2_full"
    if args.subset:
        questions = stratified_subset(questions, args.subset)
    run(questions, args.out, tag=tag,
        abstain=(args.bm25_min, args.cos_min), limit=args.limit or None)


if __name__ == "__main__":
    main()
