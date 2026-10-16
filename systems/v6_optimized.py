"""V6「优化旗舰」—— 在 V5 之上三项读取端优化（全部来自 V5 全量的逐题归因）：

1. 稠密通道升级：智谱 embedding-3（1024 维，逐文本磁盘缓存）替换 bge-small
   —— V2 已证明稠密通道在 GLM 口径下 +3pt，更强向量预期放大该收益；
2. 路由答案核验（route-verify）：V5 的 abs 损失来自 glm-5.3 旗舰"过度自信不走弃权"
   （ku-abs 1.0→0.5）。修复：路由题的旗舰答案交回 flash 核验"证据是否支持"，
   不支持则回退弃权短语（不使用任何题目侧信息，纯证据一致性检查）；
3. preference 个性化读取：V5 ss-preference 仅 0.733——该题型 judge 按 rubric
   评"是否利用了用户个人信息"，短句式太薄。改为允许两句话、显式编织用户已有物品/习惯。

写入端与 V3/V5 完全一致（原子抽取走磁盘缓存，零重复成本）。
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
    GEN_SYSTEM_V3, AtomStore, _dedupe_atoms, extract_group_cached, retrieve_atoms,
)
from systems.v1_lexical import QTYPE_HINTS, deepseek_config, load_env, stratified_subset
from systems.v4_route import REASON_TYPES, ABSTAIN_PHRASE
from systems.runner import run_pipeline

PREF_HINT = ("Personalization question: give advice that EXPLICITLY builds on the user's own "
             "previously mentioned items, habits or plans (name them). At most two short sentences.")

VERIFY_SYSTEM = """You are verifying an answer against memory evidence. Decide whether the proposed answer is directly supported by the memories.
Rules:
- Supported = the memories contain the fact (or the arithmetic follows from dates in the memories).
- If the memories do not contain the asked information, the answer is NOT supported even if it sounds plausible.
- Answer with exactly one word: yes or no."""


def _build_evidence_lines(cards: list) -> list[str]:
    return [f"[{i}] ({c.valid_from}{'→' + c.valid_to + '(superseded)' if c.valid_to else ''}) {c.content}"
            for i, c in enumerate(cards, 1)]


def answer_v6(store: MemoryStore, index, backend: OpenAICompatBackend, chat_cfg: ModelConfig,
              flagship_cfg: ModelConfig | None, question: str, as_of: str | None,
              retry_min_bm25: float = 3.0) -> tuple[str, dict]:
    cards, bm25_top, info, cos_top = retrieve_atoms(store, index, question, as_of)
    meta = {"qtype": info["qtype"], "top_bm25": round(bm25_top, 2),
            "top_cos": round(cos_top, 3), "n_evidence": len(cards)}
    if not cards:
        meta["abstain"] = "score_gate"
        return "I don't have that information in my memory.", meta
    keep_atoms = info["qtype"] in ("temporal_reasoning", "knowledge_update")
    cards = _dedupe_atoms(cards, keep_atoms)
    meta["n_evidence"] = len(cards)
    meta["evidence_chars"] = sum(len(c.content) for c in cards)

    routed = flagship_cfg is not None and info["qtype"] in REASON_TYPES
    model_cfg = flagship_cfg if routed else chat_cfg
    hint = PREF_HINT if info["qtype"] == "single_session_preference" \
        else QTYPE_HINTS.get(info["qtype"], "")
    lines = _build_evidence_lines(cards)
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

    # 路由答案核验：旗舰答案须通过 flash 的证据一致性检查（防过度自信的幻觉，
    # 尤其保护 unanswerable 题；不使用任何题目侧信息）
    if routed and ans and ans.lower().strip(". !") != ABSTAIN_PHRASE:
        focus = cards[:10]
        vlines = _build_evidence_lines(focus)
        vuser = (f"Question: {question}\n\nProposed answer: {ans}\n\n"
                 "Memories:\n" + "\n".join(vlines) +
                 "\n\nIs the proposed answer directly supported by the memories? yes or no.")
        compv = backend.complete(task="verify", model_cfg=chat_cfg, system=VERIFY_SYSTEM,
                                 user=vuser, extra={"question": question})
        verdict = (compv.text or "").strip().lower()
        uv = compv.usage or {}
        usage["prompt_tokens"] += uv.get("prompt_tokens", 0)
        usage["completion_tokens"] += uv.get("completion_tokens", 0)
        meta["verification"] = "confirmed" if verdict.startswith("yes") else \
            ("rejected" if verdict.startswith("no") else "unclear:" + verdict[:12])
        if verdict.startswith("no"):
            meta["abstain"] = "verify_rejected"
            return "I don't have that information in my memory.", meta

    # 拒答恢复（与 V4/V5 相同：假拒答捞回，真无证据时模型会再次弃权）
    if ans.lower().strip(". !") == ABSTAIN_PHRASE and bm25_top >= retry_min_bm25:
        focus = cards[:8]
        lines2 = _build_evidence_lines(focus)
        user2 = (f"Question date: {as_of or 'unknown'}\nQuestion: {question}\n"
                 "\nMemories:\n" + "\n".join(lines2))
        comp2 = backend.complete(task="generate", model_cfg=chat_cfg,
                                 system=GEN_SYSTEM_V3 +
                                 "\n\nIMPORTANT second pass: you previously said you don't know."
                                 " Re-examine carefully; counts require enumerating items, durations"
                                 " require subtracting dates, latest date = current state. If ANY"
                                 " memory is relevant, answer from it. Keep the abstention ONLY if"
                                 " nothing relates to the question at all.",
                                 user=user2, extra={"question": question})
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


def run(questions: list[dict], out_dir: str | Path, tag: str = "v6",
        model_cfg: ModelConfig | None = None, flagship_cfg: ModelConfig | None = None,
        limit: int | None = None, workers: int = 2) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chat_cfg = model_cfg or deepseek_config()
    backend = OpenAICompatBackend()

    if limit:
        questions = questions[:limit]
    groups = group_by_haystack(questions)

    def build_group(g):
        store = MemoryStore(":memory:")
        atoms = AtomStore(store)
        extracted = extract_group_cached(backend, chat_cfg, g)
        for s, atom_dicts in zip(g["sessions"], extracted):
            atoms.apply_atoms(s, atom_dicts)
        from systems.v2_dense import DenseIndex
        index = DenseIndex()
        index.add(store.all_cards())
        return store, index

    def answer_one(ctx, q):
        store, index = ctx
        try:
            ans, meta = answer_v6(store, index, backend, chat_cfg, flagship_cfg,
                                  q["question"], q.get("question_date"))
        except Exception as e:
            ans = "I don't have that information in my memory."
            meta = {"qtype": "error", "error": str(e)[:200]}
        return ans, meta

    t0 = time.time()
    records, usage_total = run_pipeline(groups, build_group, answer_one, out_dir, tag,
                                        workers=workers)
    n_recovered = sum(1 for r in records if r.get("recovered"))
    n_rejected = sum(1 for r in records if r.get("verification") == "rejected")

    summary = {"tag": tag, "n_questions": len(records),
               "recovered_from_abstain": n_recovered,
               "verify_rejected": n_rejected,
               "elapsed_s": round(time.time() - t0, 1), "usage": usage_total,
               "qtype_dist": dict(Counter(r["question_type"] for r in records))}
    (out_dir / f"run_summary_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    import argparse
    from systems.v1_lexical import zhipu_config, zhipu_flagship_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/longmemeval/longmemeval_oracle.json")
    ap.add_argument("--out", default="runs/v6")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-flagship", action="store_true")
    args = ap.parse_args()
    load_env()
    questions = json.loads(Path(args.data).read_text(encoding="utf-8"))
    tag = "v6_sub%d" % args.subset if args.subset else "v6_full"
    if args.subset:
        questions = stratified_subset(questions, args.subset)
    flagship = None if args.no_flagship else zhipu_flagship_config()
    run(questions, args.out, tag=tag, model_cfg=zhipu_config(),
        flagship_cfg=flagship, limit=args.limit or None)


if __name__ == "__main__":
    main()
