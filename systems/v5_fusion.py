"""V5「融合旗舰」—— 双层记忆(V3) × 题型路由 + 拒答恢复(V4)。

写入端 = V3：LLM 记忆原子(语义层) + 规则信息句/助手笔记(情景层保底)，
         冲突失效、容量遗忘、原子引文去重；
读取端 = V4：temporal/multi_session/knowledge_update → deepseek-reasoner，
         假拒答 two-pass 恢复（信号阈值保护真拒答）。

两个机制正交（V3 改写入、V4 改读取），本模块只做组合，不引入新变量。
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

from systems.v3_layered import (
    GEN_SYSTEM_V3, AtomStore, _dedupe_atoms, extract_atoms, retrieve_atoms,
)
from systems.v1_lexical import deepseek_config, load_env, stratified_subset
from systems.v4_route import REASON_TYPES, make_reasoner_cfg, ABSTAIN_PHRASE

RETRY_ADDON = """

IMPORTANT second pass: you previously answered that you don't have the information.
Re-examine the memories carefully. The answer is usually implied by one or more atoms:
counts require enumerating matching items; durations require subtracting dates;
the atom with the latest date is the current state. If ANY memory is relevant, answer from it.
Keep the abstention ONLY if nothing relates to the question at all."""


def answer_v5(store: MemoryStore, index, backend: OpenAICompatBackend, chat_cfg: ModelConfig,
              reasoner_cfg: ModelConfig | None, question: str, as_of: str | None,
              retry_min_bm25: float = 3.0) -> tuple[str, dict]:
    cards, bm25_top, info, cos_top = retrieve_atoms(store, index, question, as_of)
    meta = {"qtype": info["qtype"], "top_bm25": round(bm25_top, 2),
            "top_cos": round(cos_top, 3)}
    if not cards:
        meta["abstain"] = "score_gate"
        return "I don't have that information in my memory.", meta
    keep_atoms = info["qtype"] in ("temporal_reasoning", "knowledge_update")
    cards = _dedupe_atoms(cards, keep_atoms)
    meta["n_evidence"] = len(cards)

    model_cfg = reasoner_cfg if (reasoner_cfg and info["qtype"] in REASON_TYPES) else chat_cfg
    lines = [f"[{i}] ({c.valid_from}{'→' + c.valid_to + '(superseded)' if c.valid_to else ''}) {c.content}"
             for i, c in enumerate(cards, 1)]
    from systems.v1_lexical import QTYPE_HINTS
    hint = QTYPE_HINTS.get(info["qtype"], "")
    user = (f"Question date: {as_of or 'unknown'}\nQuestion: {question}\n"
            + (f"Hint: {hint}\n" if hint else "")
            + "\nMemories:\n" + ("\n".join(lines) if lines else "(none)"))
    comp = backend.complete(task="generate", model_cfg=model_cfg, system=GEN_SYSTEM_V3,
                            user=user, extra={"question": question})
    ans = (comp.text or "").strip()
    u = comp.usage or {}
    usage = {"prompt_tokens": u.get("prompt_tokens", 0),
             "completion_tokens": u.get("completion_tokens", 0)}
    meta["model"] = model_cfg.name

    if ans.lower().strip(". !") == ABSTAIN_PHRASE and bm25_top >= retry_min_bm25:
        focus = cards[:8]
        lines2 = [f"[{i}] ({c.valid_from}{'→' + c.valid_to if c.valid_to else ''}) {c.content}"
                  for i, c in enumerate(focus, 1)]
        user2 = (f"Question date: {as_of or 'unknown'}\nQuestion: {question}\n"
                 "\nMemories:\n" + "\n".join(lines2))
        comp2 = backend.complete(task="generate", model_cfg=chat_cfg,
                                 system=GEN_SYSTEM_V3 + RETRY_ADDON, user=user2,
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


def run(questions: list[dict], out_dir: str | Path, tag: str = "v5",
        model_cfg: ModelConfig | None = None, use_reasoner: bool = True,
        reasoner_cfg: ModelConfig | None = None,
        limit: int | None = None, workers: int = 3) -> dict:
    from systems.runner import run_pipeline
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chat_cfg = model_cfg or deepseek_config()
    if not use_reasoner:
        reasoner_cfg = None
    elif reasoner_cfg is None:
        reasoner_cfg = make_reasoner_cfg(chat_cfg)
    backend = OpenAICompatBackend()

    if limit:
        questions = questions[:limit]
    groups = group_by_haystack(questions)

    def build_group(g):
        store = MemoryStore(":memory:")
        atoms = AtomStore(store)
        # 抽取禁用思考（同 V3）：提速且防 JSON 截断
        import dataclasses
        extract_fast = dataclasses.replace(chat_cfg, thinking=False)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            extracted = list(ex.map(lambda s: extract_atoms(backend, extract_fast, s),
                                    g["sessions"]))
        for s, atom_dicts in zip(g["sessions"], extracted):
            atoms.apply_atoms(s, atom_dicts)
        from systems.v2_dense import DenseIndex
        index = DenseIndex()
        index.add(store.all_cards())
        return store, index

    def answer_one(ctx, q):
        store, index = ctx
        try:
            ans, meta = answer_v5(store, index, backend, chat_cfg, reasoner_cfg,
                                  q["question"], q.get("question_date"))
        except Exception as e:
            ans = "I don't have that information in my memory."
            meta = {"qtype": "error", "error": str(e)[:200]}
        return ans, meta

    t0 = time.time()
    records, usage_total = run_pipeline(
        groups, build_group, answer_one, out_dir, tag, workers=workers)
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
    ap.add_argument("--out", default="runs/v5")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-reasoner", action="store_true")
    args = ap.parse_args()
    load_env()
    questions = json.loads(Path(args.data).read_text(encoding="utf-8"))
    tag = "v5_sub%d" % args.subset if args.subset else "v5_full"
    if args.subset:
        questions = stratified_subset(questions, args.subset)
    run(questions, args.out, tag=tag, use_reasoner=not args.no_reasoner,
        limit=args.limit or None)


if __name__ == "__main__":
    main()
