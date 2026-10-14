"""V4「路由 + 拒答恢复」——在 V1.2 词法核心上做读取端迭代。

两个机制（失败模式驱动，见 ITERATIONS.md）：
1. 题型路由：temporal/multi_session/knowledge_update 推理重 → deepseek-reasoner；
   其余 → deepseek-chat（快、省）。
2. 拒答恢复：模型自评弃权且检索信号非零时，用"禁止弃权"提示重问一次
   （聚焦 top-8 证据），把假拒答捞回来；真拒答（信号为零）不重试，保住 _abs 分。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from lmem.config import ModelConfig
from lmem.evals.longmemeval import group_by_haystack
from lmem.llm.client import OpenAICompatBackend
from lmem.memory.store import MemoryStore

from systems.v1_lexical import (
    QTYPE_HINTS, GEN_SYSTEM, analyze_query_rules, build_prompt, deepseek_config,
    ingest_rules, load_env, retrieve, stratified_subset,
)

import os

REASONER_MODEL = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")
REASON_TYPES = {"temporal_reasoning", "multi_session", "knowledge_update"}
ABSTAIN_PHRASE = "i don't have that information in my memory"

RETRY_SYSTEM = GEN_SYSTEM + """

IMPORTANT second pass: you previously answered that you don't have the information.
Re-examine the memories carefully. The answer is usually implied by one or more snippets:
counts require enumerating matching items; durations require subtracting dates; "current"
values are the ones with the latest date. If ANY snippet is relevant, answer from it.
Keep the abstention ONLY if nothing relates to the question at all."""


def make_reasoner_cfg(base: ModelConfig) -> ModelConfig:
    import dataclasses
    return dataclasses.replace(base, name=REASONER_MODEL, max_tokens=3000)


def answer_with_recovery(store: MemoryStore, backend: OpenAICompatBackend,
                         chat_cfg: ModelConfig, reasoner_cfg: ModelConfig | None,
                         question: str, as_of: str | None,
                         retry_min_bm25: float = 3.0) -> tuple[str, dict]:
    cards, top_score, info = retrieve(store, question, as_of)
    meta = {"qtype": info["qtype"], "top_bm25": round(top_score, 2), "n_evidence": len(cards)}
    if not cards or top_score < 0.5:
        meta["abstain"] = "score_gate"
        return "I don't have that information in my memory.", meta

    model_cfg = reasoner_cfg if (reasoner_cfg and info["qtype"] in REASON_TYPES) else chat_cfg
    comp = backend.complete(task="generate", model_cfg=model_cfg, system=GEN_SYSTEM,
                            user=build_prompt(cards, question, as_of, info["qtype"]),
                            extra={"question": question})
    ans = (comp.text or "").strip()
    u = comp.usage or {}
    usage = {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0)}
    meta["model"] = model_cfg.name

    # 拒答恢复：假拒答捞回（真无证据时模型会再次弃权）
    if ans.lower().strip(". !") == ABSTAIN_PHRASE and top_score >= retry_min_bm25:
        focus = cards[:8]
        comp2 = backend.complete(task="generate", model_cfg=chat_cfg, system=RETRY_SYSTEM,
                                 user=build_prompt(focus, question, as_of, info["qtype"]),
                                 extra={"question": question})
        ans2 = (comp2.text or "").strip()
        u2 = comp2.usage or {}
        usage["prompt_tokens"] += u2.get("prompt_tokens", 0)
        usage["completion_tokens"] += u2.get("completion_tokens", 0)
        meta["retried"] = True
        if ans2.lower().strip(". !") != ABSTAIN_PHRASE:
            ans = ans2
            meta["recovered"] = True
    meta["usage"] = usage
    return ans or "I don't have that information in my memory.", meta


def run(questions: list[dict], out_dir: str | Path, tag: str = "v4",
        model_cfg: ModelConfig | None = None, use_reasoner: bool = True,
        limit: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chat_cfg = model_cfg or deepseek_config()
    reasoner_cfg = make_reasoner_cfg(chat_cfg) if use_reasoner else None
    backend = OpenAICompatBackend()

    if limit:
        questions = questions[:limit]
    groups = group_by_haystack(questions)

    hyp_path = out_dir / f"hyp_{tag}.jsonl"
    records: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()
    with hyp_path.open("w", encoding="utf-8") as f:
        for gi, g in enumerate(groups, 1):
            store = MemoryStore(":memory:")
            for s in g["sessions"]:
                ingest_rules(store, s)
            for q in g["questions"]:
                ans, meta = answer_with_recovery(store, backend, chat_cfg, reasoner_cfg,
                                                 q["question"], q.get("question_date"))
                f.write(json.dumps({"question_id": q["question_id"], "hypothesis": ans},
                                   ensure_ascii=False) + "\n")
                u = meta.pop("usage", {}) or {}
                usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += u.get("completion_tokens", 0)
                records.append({"question_id": q["question_id"],
                                "question_type": q.get("question_type"),
                                "is_abs": q["question_id"].endswith("_abs"),
                                "answer": q.get("answer", ""), "hypothesis": ans, **meta})
            if gi % 25 == 0 or gi == len(groups):
                print(f"[{tag}] 组 {gi}/{len(groups)} 已答={len(records)} "
                      f"用时={time.time()-t0:.0f}s", flush=True)
            store.close()

    (out_dir / f"records_{tag}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    n_recovered = sum(1 for r in records if r.get("recovered"))
    summary = {"tag": tag, "n_questions": len(records),
               "recovered_from_abstain": n_recovered,
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
    ap.add_argument("--out", default="runs/v4")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-reasoner", action="store_true")
    args = ap.parse_args()
    load_env()
    questions = json.loads(Path(args.data).read_text(encoding="utf-8"))
    tag = "v4_sub%d" % args.subset if args.subset else "v4_full"
    if args.subset:
        questions = stratified_subset(questions, args.subset)
    run(questions, args.out, tag=tag, use_reasoner=not args.no_reasoner,
        limit=args.limit or None)


if __name__ == "__main__":
    main()
