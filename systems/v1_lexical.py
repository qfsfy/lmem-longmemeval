"""V1「词法笔记」记忆系统 —— SQLite + 规则写入 + BM25 检索 + LLM 短答案生成。

设计定位（迭代一）：
- 把"检索质量"隔离为唯一变量：写入零 LLM、检索零向量，答案生成是唯一模型调用；
- 记忆单元 = 信息句（启发式过滤寒暄），保留原文保证可溯源；
- 身份类卡（姓名/居住/职业/生日/偏好）走 profile 直取 + 冲突失效（双时间轴）；
- 拒答 = BM25 相对信号（无命中/低分）+ 模型自评双通道。

技术栈选型理由（为什么 / 为什么不）见 ITERATIONS.md 迭代一。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from lmem.config import ModelConfig
from lmem.evals.longmemeval import _to_sessions, group_by_haystack
from lmem.llm.client import BaseBackend, OpenAICompatBackend
from lmem.memory.schema import MemoryCard, Session
from lmem.memory.store import MemoryStore

# --------------------------------------------------------------------------
# 写入：英文信息性启发式（零 LLM）
# --------------------------------------------------------------------------

FILLER_RE = re.compile(
    r"^(hi+|hello+|hey|thanks|thank you( so much| very much)?|ok(ay)?|sure|yes|yeah|yep|"
    r"no|nope|got it|sounds good|awesome|great|cool|nice|wow|oh+|i see|alright|please|"
    r"sorry|excuse me|good (morning|afternoon|evening)|of course|you'?re welcome|bye+|"
    r"goodbye|that'?s (great|good|nice|cool|awesome)|haha+|lol)[\s!,.?]*$",
    re.I,
)
FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|i'll|i'd|my|me|mine|myself|we|our|us)\b", re.I)
ASSISTEE_RE = re.compile(r"\b(you|your|you're|you're)\b", re.I)
PREF_RE = re.compile(
    r"\b(love[sd]?|like[sd]?|enjoy[sd]?|prefer[sd]?|favorit\w*|hate[sd]?|dislike[sd]?|"
    r"fond of|interested in|fan of)\b", re.I)
IDENTITY_RE = re.compile(r"\b(my name is|i am called|call me|my name's)\s+([a-z][\w'-]{1,30})", re.I)
RESIDENCE_RE = re.compile(
    r"\bi (?:live|lived|am living|used to live|moved|am moving|relocat\w+|settled|am based)\s*"
    r"(?:down )?(?:in|to|near|at)\s+([a-z][\w .'-]{1,40})", re.I)
OCCUPATION_RE = re.compile(
    r"\bi (?:am|'m|was|work(?:ed)? as|work(?:ed)? (?:at|for|in)|currently work)|"
    r"\bmy job(?: is)?|i (?:study|studied|teach(?:es)?)|my major(?: is)?", re.I)
BIRTHDAY_RE = re.compile(r"\b(my birthday|i was born)\b", re.I)
EVENT_RE = re.compile(
    r"\b(went|visit\w+|moved|started|finish\w+|graduat\w+|travel\w*|trip|planned|booked|"
    r"attended|won|met|joined|bought|sold|adopted|retired|celebrat\w+|anniversary|"
    r"had (?:dinner|lunch|breakfast)|cooked|baked|watched|read|finished)\b", re.I)
NUM_RE = re.compile(r"\d")

MAX_UNIT_WORDS = 60


def _sentences(text: str) -> list[str]:
    # 同时按行切（markdown 列表项），短碎片合并进前一块，避免结构性枚举被切碎
    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    parts: list[str] = []
    for p in (x.strip() for x in raw):
        if not p:
            continue
        if len(p.split()) < 3 and parts and not re.search(r"\d", p):
            parts[-1] += " " + p
        else:
            parts.append(p)
    return parts


def _classify(sentence: str) -> tuple[str, str, str]:
    """返回 (predicate, value, kind)。value 仅对身份类有效（用于冲突检测）。"""
    m = IDENTITY_RE.search(sentence)
    if m:
        return "name", m.group(2).strip(), "profile"
    m = RESIDENCE_RE.search(sentence)
    if m:
        return "lives_in", m.group(1).strip(), "profile"
    if OCCUPATION_RE.search(sentence):
        return "job", "", "profile"
    if BIRTHDAY_RE.search(sentence):
        return "birthday", "", "profile"
    if PREF_RE.search(sentence):
        return "likes", "", "profile"
    return "other", "", "fact"


def _informative(sentence: str, role: str) -> bool:
    if FILLER_RE.match(sentence):
        return False
    n_words = len(sentence.split())
    if n_words < 3 and not NUM_RE.search(sentence):
        return False
    if NUM_RE.search(sentence) or PREF_RE.search(sentence) or EVENT_RE.search(sentence):
        return True
    if role == "user":
        return bool(FIRST_PERSON_RE.search(sentence)) and n_words >= 4
    return bool(ASSISTEE_RE.search(sentence)) and n_words >= 5


def extract_cards_rules(session: Session) -> list[MemoryCard]:
    """会话 → 记忆卡：按轮次聚合信息句，一轮最多一张卡（保留共现上下文）。"""
    cards: list[MemoryCard] = []
    for ti, turn in enumerate(session.turns):
        sents = [s for s in _sentences(turn.content) if _informative(s, turn.role)]
        if not sents:
            continue
        units: list[list[str]] = [[]]
        for s in sents:
            if sum(len(x.split()) for x in units[-1]) + len(s.split()) > MAX_UNIT_WORDS:
                units.append([])
            units[-1].append(s)
        for chunk in units:
            if not chunk:
                continue
            content = " ".join(chunk)
            pred, value, kind = _classify(content)
            cards.append(MemoryCard(
                kind="note" if turn.role == "assistant" else kind,
                content=content,
                subject="user",
                predicate=pred,
                value=value,
                valid_from=session.date,
                observed_at=session.date,
                session_id=session.session_id,
                turn_start=ti,
                turn_end=ti,
                confidence=0.9,
            ))
    return cards


_PROFILE_PREDS = ("name", "lives_in", "job", "birthday", "likes")


def ingest_rules(store: MemoryStore, session: Session) -> dict:
    """规则写入 + 身份类冲突失效（保留历史版本，支持 as-of 查询）。"""
    cards = extract_cards_rules(session)
    superseded = 0
    for new in cards:
        if new.predicate in _PROFILE_PREDS:
            for old in store.active_at(session.date):
                if old.predicate == new.predicate and old.id != new.id:
                    if new.predicate in ("name", "lives_in") and old.value and new.value \
                            and old.value.lower() != new.value.lower():
                        store.supersede(old.id, new.id, session.date)
                        superseded += 1
    for c in cards:
        store.add_card(c)
    store.add_session(session.session_id, session.date, len(session.turns))
    return {"cards_added": len(cards), "superseded": superseded}


# --------------------------------------------------------------------------
# 检索：规则查询分析 + BM25 + 画像直取
# --------------------------------------------------------------------------

_UPDATE_RE = re.compile(r"\b(used to|use to|no longer|switch\w*|changed?|now|these days|currently)\b", re.I)
_TEMPORAL_RE = re.compile(
    r"\bhow (long|much time|many (days|weeks|months|years))\b|\bhow many days\b|"
    r"\b(when (was|did)|what date|ago|how old)\b", re.I)
_MULTI_RE = re.compile(
    r"\b(list|all (of )?the|every|how many|what (are|were)|which (books|movies|dishes|places|"
    r"people|things|activities))\b", re.I)
_PAST_STATE_RE = re.compile(r"\b(before|used to|at that time|back then|previously|first|at the time)\b", re.I)


def analyze_query_rules(question: str) -> dict:
    ql = question.lower()
    if _UPDATE_RE.search(ql):
        qtype = "knowledge_update"
    elif _TEMPORAL_RE.search(ql):
        qtype = "temporal_reasoning"
    elif _MULTI_RE.search(ql):
        qtype = "multi_session"
    elif re.search(r"\b(favorite|prefer|like best|love most)\b", ql):
        qtype = "single_session_preference"
    else:
        qtype = "single_session_user"
    return {"qtype": qtype, "include_past": bool(_PAST_STATE_RE.search(ql))}


def retrieve(store: MemoryStore, question: str, as_of: str | None,
             bm25_top_k: int = 24, profile_max: int = 5) -> tuple[list[MemoryCard], float, dict]:
    info = analyze_query_rules(question)
    date = None
    if as_of and not info["include_past"] and info["qtype"] != "knowledge_update":
        date = as_of  # 只看当时仍有效的记忆；追问过去状态或更新类题型放开时间旅行
    pool = store.all_cards() if date is None else store.active_at(date)
    allowed = {c.id for c in pool}
    # 聚合类题型：小库（oracle 干草堆）直接全量读取，保证枚举完整；大库退回 top-k
    probe = store.search(question, k=1, date=date)
    top_score = probe[0][1] if probe else 0.0
    if info["qtype"] == "multi_session" and len(pool) <= 200:
        scored = [(c, 0.0) for c in pool]
    else:
        k = bm25_top_k + 12 if info["qtype"] == "multi_session" else bm25_top_k
        scored = store.search(question, k=k, date=date)
    cards = [c for c, _ in scored]
    existing = {c.id for c in cards}
    if date is not None:
        profiles = [c for c in store.active_at(date) if c.kind == "profile"]
    else:
        profiles = [c for c in store.all_cards() if c.kind == "profile"]
    profiles.sort(key=lambda c: c.observed_at, reverse=True)
    cards += [c for c in profiles[:profile_max] if c.id not in existing]
    return cards, top_score, info


# --------------------------------------------------------------------------
# 答案生成（唯一 LLM 调用）
# --------------------------------------------------------------------------

GEN_SYSTEM = """You are a personal memory assistant. Answer the user's question using ONLY the numbered memory snippets.

Procedure:
- Scan EVERY snippet for relevant facts. User statements and assistant statements are both evidence.
- Counting/aggregation questions: first ENUMERATE every matching item you find (with its snippet number), then count or sum them.
- Conflicting facts (e.g., a number that changed over time): the snippet with the LATEST date reflects the current state, unless the question asks about the past.
- Duration/ordering questions: compute from the dates attached to snippets and the given question date.

Answer style:
- ONE short sentence (<= 35 words) directly containing the key fact, in the language of the question. No explanations, no citations.
- Personal questions (advice/preferences): weave in the user's own previously mentioned items or habits from the snippets.
- Only if you scanned all snippets and the needed information is truly absent, reply exactly: I don't have that information in my memory."""


QTYPE_HINTS = {
    "multi_session": "This question likely spans several sessions: enumerate all matching items across snippets before answering.",
    "temporal_reasoning": "Compute dates/durations/ordering carefully from snippet dates and the question date. Show the key item and its date in your answer.",
    "knowledge_update": "The user's facts changed over time: give the CURRENT state (latest date), which the question asks for.",
    "single_session_preference": "Personalize: explicitly build on the user's own mentioned preferences, possessions or habits.",
}


def build_prompt(cards: list[MemoryCard], question: str, as_of: str | None,
                 qtype: str = "") -> str:
    lines = [f"[{i}] ({c.valid_from}{'→' + c.valid_to if c.valid_to else ''}) {c.content}"
             for i, c in enumerate(cards, 1)]
    hint = QTYPE_HINTS.get(qtype, "")
    return (f"Question date: {as_of or 'unknown'}\n"
            f"Question: {question}\n"
            + (f"Hint: {hint}\n" if hint else "")
            + "\nMemories:\n" + ("\n".join(lines) if lines else "(none)"))


def answer_question(store: MemoryStore, backend: BaseBackend, model_cfg: ModelConfig,
                    question: str, as_of: str | None,
                    abstain_score: float) -> tuple[str, dict]:
    cards, top_score, info = retrieve(store, question, as_of)
    meta = {"qtype": info["qtype"], "top_bm25": round(top_score, 2), "n_evidence": len(cards)}
    if not cards or top_score < abstain_score:
        meta["abstain"] = "score_gate"
        return "I don't have that information in my memory.", meta
    comp = backend.complete(
        task="generate", model_cfg=model_cfg, system=GEN_SYSTEM,
        user=build_prompt(cards, question, as_of, info["qtype"]), extra={"question": question},
    )
    ans = (comp.text or "").strip() or "I don't have that information in my memory."
    meta["usage"] = comp.usage
    meta["latency_ms"] = comp.latency_ms
    return ans, meta


# --------------------------------------------------------------------------
# 评测流水线
# --------------------------------------------------------------------------

def run(questions: list[dict], out_dir: str | Path, tag: str = "v1",
        model_cfg: ModelConfig | None = None, abstain_score: float = 0.0,
        limit: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = model_cfg or deepseek_config()
    backend = OpenAICompatBackend()

    if limit:
        questions = questions[:limit]
    groups = group_by_haystack(questions)

    hyp_path = out_dir / f"hyp_{tag}.jsonl"
    rec_path = out_dir / f"records_{tag}.json"
    records: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    t_start = time.time()

    with hyp_path.open("w", encoding="utf-8") as f:
        for gi, g in enumerate(groups, 1):
            store = MemoryStore(":memory:")
            stats = [ingest_rules(store, s) for s in g["sessions"]]
            n_cards = sum(s["cards_added"] for s in stats)
            for q in g["questions"]:
                try:
                    ans, meta = answer_question(
                        store, backend, model_cfg, q["question"],
                        as_of=q.get("question_date"), abstain_score=abstain_score,
                    )
                except Exception as e:  # 单题故障不拖垮整轮（记录后按拒答输出）
                    ans = "I don't have that information in my memory."
                    meta = {"qtype": "error", "error": str(e)[:200]}
                f.write(json.dumps({"question_id": q["question_id"], "hypothesis": ans},
                                   ensure_ascii=False) + "\n")
                u = meta.pop("usage", {}) or {}
                usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += u.get("completion_tokens", 0)
                records.append({
                    "question_id": q["question_id"], "question_type": q.get("question_type"),
                    "is_abs": q["question_id"].endswith("_abs"),
                    "answer": q.get("answer", ""), "hypothesis": ans, **meta,
                })
            if gi % 20 == 0 or gi == len(groups):
                print(f"[{tag}] 组 {gi}/{len(groups)} 卡片累计示例={n_cards} "
                      f"已答={len(records)} 用时={time.time()-t_start:.0f}s", flush=True)
            store.close()

    rec_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {
        "tag": tag, "n_questions": len(records),
        "elapsed_s": round(time.time() - t_start, 1),
        "usage": usage_total,
        "qtype_dist": dict(Counter(r["question_type"] for r in records)),
    }
    (out_dir / f"run_summary_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def deepseek_config() -> ModelConfig:
    import os
    return ModelConfig(
        name=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key_env="DEEPSEEK_API_KEY",
        temperature=0.0,
        max_tokens=120,
    )


def local_config() -> ModelConfig:
    """本地 llama.cpp server（OpenAI 兼容），零 API 成本的统一评测后端。

    启动：.tools/llama.cpp/llama-server.exe -m .models/qwen3-4b-instruct-2507-q4km.gguf \
        -ngl 99 -c 8192 --host 127.0.0.1 --port 8321
    """
    import os
    return ModelConfig(
        name=os.environ.get("LOCAL_MODEL", "qwen3-4b-instruct-2507"),
        base_url=os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8321/v1"),
        api_key_env="LOCAL_API_KEY",  # 未设置时客户端回退 "EMPTY"
        temperature=0.0,
        max_tokens=200,
    )


def load_env(env_path: str | Path = ".env") -> None:
    for line in Path(env_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            import os
            os.environ.setdefault(k.strip(), v.strip())


def stratified_subset(questions: list[dict], n: int, seed: int = 42) -> list[dict]:
    """按 question_type 分层等比抽样（固定 seed 可复现）。"""
    import random
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for q in questions:
        by_type.setdefault(q["question_type"], []).append(q)
    picked: list[dict] = []
    total = len(questions)
    for t, qs in sorted(by_type.items()):
        k = round(n * len(qs) / total)
        picked.extend(rng.sample(qs, min(k, len(qs))))
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="V1 词法记忆系统 · LongMemEval Oracle 评测")
    ap.add_argument("--data", default="data/longmemeval/longmemeval_oracle.json")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--subset", type=int, default=0, help="分层抽样题数，0=全量500")
    ap.add_argument("--limit", type=int, default=0, help="取前 N 题（调试用）")
    ap.add_argument("--abstain-score", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    load_env()
    questions = json.loads(Path(args.data).read_text(encoding="utf-8"))
    tag = args.tag or ("v1_sub%d" % args.subset if args.subset else "v1_full")
    if args.subset:
        questions = stratified_subset(questions, args.subset, args.seed)
    run(questions, args.out, tag=tag, abstain_score=args.abstain_score,
        limit=args.limit or None)


if __name__ == "__main__":
    main()
