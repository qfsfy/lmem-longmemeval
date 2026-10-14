"""V3「分层记忆」系统 —— LLM 结构化抽取 + 时间感知 + 冲突更新 + 遗忘策略。

相对 V1/V2 的机制升级（对应手册 B.1/B.2 五能力）：
- Accumulate: LLM 抽取"记忆原子"（谓词/宾语/逐字引文/日期/重要度），不存原始对话；
- Update:    同谓词不同值 → 新原子生效、旧原子失效（双时间轴版本化，可 as-of 查询）；
- Retrieve:  混合检索（BM25 + 稠密 RRF）作用于原子文本 "predicate: object — quote"，
             画像直取、聚合题型小库全读；
- Reuse:     证据按 (日期, 有效性) 结构化呈现，配时序/聚合读取协议；
- Forget:    容量上限 × (重要度 × 时效衰减 × 访问频次) 的淘汰策略（oracle 档难触发，
             用单元测试验证机制真实存在，见 tests/test_v3_forget.py）。

与 V1/V2 的对照变量只有一个：写入端从"规则信息句"换成"LLM 记忆原子"。
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from lmem.config import ModelConfig
from lmem.evals.longmemeval import group_by_haystack
from lmem.llm.client import OpenAICompatBackend
from lmem.memory.schema import MemoryCard, Session
from lmem.memory.store import MemoryStore

from systems.v1_lexical import QTYPE_HINTS, analyze_query_rules, deepseek_config, \
    extract_cards_rules, load_env, stratified_subset
from systems.v2_dense import DenseIndex

# --------------------------------------------------------------------------
# LLM 结构化抽取
# --------------------------------------------------------------------------

class Atom(BaseModel):
    predicate: str = Field(description="normalized predicate, e.g. lives_in / traveled_to / bought / job / likes")
    object: str = Field(description="object or value of the fact, e.g. Japan / Austin / software engineer")
    kind: str = Field(description="profile | event | preference | opinion | other")
    date: str = Field(description="YYYY-MM-DD when the fact became/becomes true; empty if unknown")
    quote: str = Field(description="verbatim quote (<= 30 words) from the conversation supporting this fact")
    importance: float = Field(description="0-1: how important to remember long-term")


class AtomOut(BaseModel):
    atoms: list[Atom] = Field(default_factory=list)


EXTRACT_SYSTEM = """You are a memory-writing engine. From one chat session, extract durable memory atoms about the USER (facts, events, preferences, possessions, relationships, plans, opinions they stated). Rules:
1. Only durable, reusable information. Skip greetings, small talk, pure assistant suggestions the user never engaged with.
2. quote MUST be a verbatim substring of the conversation (<= 30 words). Never paraphrase inside quote.
3. date: use an explicit date mentioned in the conversation (resolve relative dates using the session date); otherwise use the session date for events, or empty for timeless attributes.
4. predicate: short snake_case (lives_in, job, birthday, traveled_to, bought, finished_reading, likes, dislikes, has_pet, visited, attended, plans_to, ...).
5. If the user UPDATES an earlier fact (moved cities, changed job, new follower count), extract the new state as an atom with its date — the store handles supersession.
6. importance: identity/living-situation/relationships 0.9; significant events with dates 0.7-0.8; casual preferences 0.4-0.5; trivia 0.2.
7. Extract at most 12 atoms per session; prefer fewer, higher-quality ones."""


def _lenient_atoms(text: str) -> list[dict] | None:
    """宽松解析：pydantic 严格校验失败时手工容错（类型强转、缺字段补默认）。"""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    raw = data.get("atoms", [])
    if not isinstance(raw, list):
        return None
    out = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        try:
            imp = float(a.get("importance", 0.5))
        except (TypeError, ValueError):
            imp = 0.5
        out.append({
            "predicate": re.sub(r"\s+", "_", str(a.get("predicate", "")).strip().lower())[:40],
            "object": str(a.get("object", "")).strip()[:120],
            "kind": a.get("kind", "other"),
            "date": str(a.get("date", "")).strip(),
            "quote": str(a.get("quote", "")).strip()[:300],
            "importance": max(0.0, min(1.0, imp)),
        })
    return out


def extract_atoms(backend: OpenAICompatBackend, model_cfg: ModelConfig,
                  session: Session) -> list[dict]:
    dialogue = "\n".join(f"[{t.role}] {t.content[:1500]}" for t in session.turns)
    comp = backend.complete(
        task="extract", model_cfg=model_cfg, system=EXTRACT_SYSTEM,
        user=f"Session date: {session.date}\n\nConversation:\n{dialogue}",
        schema=AtomOut, extra={"date": session.date},
    )
    atoms: list[dict] | None = None
    if comp.data is not None:
        atoms = [a.model_dump() for a in comp.data.atoms]
    else:
        atoms = _lenient_atoms(comp.text)  # 严格校验失败 → 宽松解析兜底
    if not atoms:
        return []
    out = []
    for a in atoms:
        date = a["date"].strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            date = session.date if a.get("kind") == "event" else ""
        out.append({
            "predicate": a["predicate"],
            "object": a["object"],
            "kind": a["kind"] if a["kind"] in ("profile", "event", "preference", "opinion") else "other",
            "date": date,
            "quote": a["quote"],
            "importance": a["importance"],
        })
    return [a for a in out if a["object"]]


# --------------------------------------------------------------------------
# 原子库：写入 + 去重 + 冲突失效 + 遗忘
# --------------------------------------------------------------------------

_META_DDL = "CREATE TABLE IF NOT EXISTS atom_meta(id TEXT PRIMARY KEY, importance REAL, access_count INTEGER DEFAULT 0)"
_CONFLICT_PREDS_RE = re.compile(r"^(lives_in|job|has_|follower|subscriber|weight|salary)", re.I)


def atom_text(a: dict) -> str:
    q = f' — "{a["quote"]}"' if a["quote"] else ""
    return f"{a['predicate']}: {a['object']}{q}"


class AtomStore:
    """MemoryStore 之上的一层原子语义：去重、冲突失效、遗忘。"""

    CAPACITY = 400  # 单用户原子容量上限（遗忘策略触发条件）

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        store.conn.execute(_META_DDL)

    def add_atom(self, a: dict, session: Session) -> str | None:
        text = atom_text(a)
        # 去重：同谓词同宾语（忽略大小写）且已生效 → 只抬重要度
        for old in self.store.active_at(session.date):
            if old.predicate == a["predicate"] and old.value.lower() == a["object"].lower():
                self.store.conn.execute(
                    "UPDATE atom_meta SET importance=MIN(1.0, importance+0.05) WHERE id=?",
                    (old.id,))
                self.store.conn.commit()
                return None
        card = MemoryCard(
            kind="profile" if a["kind"] == "profile" else "atom",
            content=text, subject="user",
            predicate=a["predicate"], value=a["object"],
            valid_from=a["date"] or session.date,
            observed_at=session.date,
            session_id=session.session_id, confidence=0.9,
        )
        self.store.add_card(card)
        self.store.conn.execute(
            "INSERT OR REPLACE INTO atom_meta VALUES (?,?,0)", (card.id, a["importance"]))
        self.store.conn.commit()
        # 冲突失效：同谓词、值不同、且谓词属于"状态类"（有唯一当前值）
        if _CONFLICT_PREDS_RE.match(a["predicate"]) or a["predicate"] in ("lives_in", "job"):
            for old in self.store.active_at(session.date):
                if old.id == card.id or old.predicate != a["predicate"]:
                    continue
                if old.value.lower() != a["object"].lower():
                    self.store.supersede(old.id, card.id, session.date)
        self._enforce_capacity(session.date)
        return card.id

    # -------- Forget：容量 × (重要度 × 时效衰减 × log 访问) 淘汰 --------
    def _enforce_capacity(self, today: str) -> int:
        cards = self.store.all_cards()
        if len(cards) <= self.CAPACITY:
            return 0
        meta = {r[0]: (r[1], r[2]) for r in
                self.store.conn.execute("SELECT id, importance, access_count FROM atom_meta")}
        def evict_score(c: MemoryCard) -> float:
            imp, acc = meta.get(c.id, (0.5, 0))
            try:
                from datetime import date as _d
                days = max(0, (_d.fromisoformat(today) - _d.fromisoformat(c.observed_at)).days)
            except ValueError:
                days = 0
            recency = math.exp(-days / 180.0)
            access = min(1.0, math.log1p(acc) / math.log1p(10))
            # 加性混合：重要度始终占主导，时效与访问做修正（乘法形式在 acc=0 时会坍缩为 0）
            return imp * (0.7 * recency + 0.3 * access)
        ranked = sorted(cards, key=evict_score)
        evicted = 0
        for c in ranked[: len(cards) - self.CAPACITY]:
            self.store.conn.execute("DELETE FROM cards WHERE id=?", (c.id,))
            self.store.conn.execute("DELETE FROM atom_meta WHERE id=?", (c.id,))
            evicted += 1
        self.store.conn.commit()
        self.store._dirty = True
        return evicted

    def ingest_session(self, backend: OpenAICompatBackend, model_cfg: ModelConfig,
                       session: Session, use_llm: bool = True) -> dict:
        added = 0
        for old in self.store.all_cards():
            if old.session_id == session.session_id:
                self.store.conn.execute("DELETE FROM cards WHERE id=?", (old.id,))
        self.store.add_session(session.session_id, session.date, len(session.turns))
        atom_dicts = extract_atoms(backend, model_cfg, session) if use_llm else []
        added = self.apply_atoms(session, atom_dicts)
        return {"atoms": added}

    def apply_atoms(self, session: Session, atom_dicts: list[dict]) -> int:
        """按会话时序应用：语义层(LLM 原子) → 情景层(规则句保底) → 冲突失效 → 容量淘汰。

        双层设计（手册 B.1 分层记忆）：原子层提供谓词/日期结构（时序/聚合题），
        情景层保留原文信息句保证召回下限不低于 V1。
        """
        added = 0
        for a in atom_dicts:
            if self.add_atom(a, session):
                added += 1
        # 情景层：V1 的规则信息句（用户轮次）+ 助手笔记
        for card in extract_cards_rules(session):
            if card.kind == "note":
                continue  # 助手内容统一走 _assistant_notes 切块
            self.store.add_card(card)
            self.store.conn.execute(
                "INSERT OR REPLACE INTO atom_meta VALUES (?,?,0)", (card.id, 0.5))
        for ti, turn in enumerate(session.turns):
            if turn.role != "assistant":
                continue
            for chunk in _assistant_notes(turn.content):
                card = MemoryCard(kind="note", content=chunk, subject="assistant",
                                  predicate="", value="", valid_from=session.date,
                                  observed_at=session.date, session_id=session.session_id,
                                  turn_start=ti, turn_end=ti, confidence=0.8)
                self.store.add_card(card)
                self.store.conn.execute(
                    "INSERT OR REPLACE INTO atom_meta VALUES (?,?,0)", (card.id, 0.3))
        self.store.conn.commit()
        return added


_NOTE_MAX_WORDS = 45

def _assistant_notes(content: str) -> list[str]:
    """助手长回复切成 ≤45 词的笔记块（markdown 列表项按行聚合）。"""
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    chunks: list[str] = []
    cur = ""
    for l in lines:
        cand = (cur + " " + l).strip()
        if len(cand.split()) > _NOTE_MAX_WORDS and cur:
            chunks.append(cur)
            cur = l
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks[:40]  # 单会话笔记上限，控制噪声


# --------------------------------------------------------------------------
# 检索 + 生成
# --------------------------------------------------------------------------

def retrieve_atoms(store: MemoryStore, index: DenseIndex, question: str,
                   as_of: str | None, bm25_top_k: int = 24):
    info = analyze_query_rules(question)
    date = None
    if as_of and not info["include_past"] and info["qtype"] != "knowledge_update":
        date = as_of
    pool = store.all_cards() if date is None else store.active_at(date)
    allowed = {c.id for c in pool}
    probe = store.search(question, k=1, date=date)
    bm25_top = probe[0][1] if probe else 0.0

    # 记忆召回预算：聚合题放宽到 48，仍按混合相关性排序（不拼塞全库/全历史）
    k = 48 if info["qtype"] == "multi_session" else bm25_top_k
    bm25_hits = store.search(question, k=k, date=date)
    dense_hits = index.search(question, allowed, k)
    scores: dict[str, float] = {}
    for rank, (card, _) in enumerate(bm25_hits):
        scores[card.id] = scores.get(card.id, 0.0) + 1.0 / (60 + rank + 1)
    for rank, (cid, _) in enumerate(dense_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    by_id = {c.id: c for c in pool}
    cards = [by_id[cid] for cid, _ in ranked if cid in by_id]
    cos_top = dense_hits[0][1] if dense_hits else 0.0
    existing = {c.id for c in cards}
    profiles = [c for c in pool if c.kind == "profile"]
    profiles.sort(key=lambda c: c.observed_at, reverse=True)
    cards += [c for c in profiles[:5] if c.id not in existing]
    return cards, bm25_top, info, cos_top


GEN_SYSTEM_V3 = """You are a personal memory assistant. Answer the user's question using ONLY the numbered memory atoms.

Procedure:
- Scan EVERY atom. Atoms look like: [i] (valid_from→valid_to) predicate: object — "quote". valid_to marks an atom no longer current (kept for history).
- Counting/aggregation: enumerate all matching atoms first, then count/sum.
- Conflicts: the atom with the LATEST valid_from is the current state; use older ones only for past-state questions.
- Durations/ordering: compute from atom dates and the question date, carefully.

Answer style:
- ONE short sentence (<= 35 words) containing the key fact, in the question's language. No citations.
- Personal questions: build on the user's own mentioned items/habits.
- Only if all atoms were scanned and the information is truly absent, reply exactly: I don't have that information in my memory."""


def _dedupe_atoms(cards: list[MemoryCard], keep_atoms: bool) -> list[MemoryCard]:
    """原子引文去重：原文句已逐字覆盖的原子不再单独入提示词（防计数双算）。

    keep_atoms=True（时序/知识更新题）时保留全部原子——它们的日期/谓词结构是关键。
    """
    if keep_atoms:
        return cards
    facts = [c for c in cards if c.kind != "atom"]
    fact_text = " \n ".join(c.content.lower() for c in facts)
    out = list(facts)
    for c in cards:
        if c.kind != "atom":
            continue
        m = re.search(r'—\s*"(.*)"\s*$', c.content, re.S)
        quote = (m.group(1) if m else c.value).lower().strip()
        if len(quote) >= 15 and quote in fact_text:
            continue  # 原子只是原文的复述 → 跳过
        out.append(c)
    return out


def answer_question(store: MemoryStore, index: DenseIndex, backend: OpenAICompatBackend,
                    model_cfg: ModelConfig, question: str, as_of: str | None) -> tuple[str, dict]:
    cards, bm25_top, info, cos_top = retrieve_atoms(store, index, question, as_of)
    meta = {"qtype": info["qtype"], "top_bm25": round(bm25_top, 2),
            "top_cos": round(cos_top, 3), "n_evidence": len(cards)}
    if not cards:
        meta["abstain"] = "score_gate"
        return "I don't have that information in my memory.", meta
    keep_atoms = info["qtype"] in ("temporal_reasoning", "knowledge_update")
    cards = _dedupe_atoms(cards, keep_atoms)
    meta["n_evidence_after_dedupe"] = len(cards)
    meta["evidence_chars"] = sum(len(c.content) for c in cards)
    lines = []
    for i, c in enumerate(cards, 1):
        span = c.valid_from + ("" if not c.valid_to else "→" + c.valid_to + "(superseded)")
        lines.append(f"[{i}] ({span}) {c.content}")
    hint = QTYPE_HINTS.get(info["qtype"], "")
    user = (f"Question date: {as_of or 'unknown'}\nQuestion: {question}\n"
            + (f"Hint: {hint}\n" if hint else "")
            + "\nMemories:\n" + ("\n".join(lines) if lines else "(none)"))
    comp = backend.complete(task="generate", model_cfg=model_cfg, system=GEN_SYSTEM_V3,
                            user=user, extra={"question": question})
    ans = (comp.text or "").strip() or "I don't have that information in my memory."
    meta["usage"] = comp.usage
    return ans, meta


# --------------------------------------------------------------------------
# 流水线
# --------------------------------------------------------------------------

_ATOM_CACHE_DIR = Path("runs/atom_cache")


def _group_cache_key(g: dict) -> str:
    import hashlib
    raw = json.dumps([[s.session_id, s.date] for s in g["sessions"]], sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _load_cached_atoms(g: dict) -> list | None:
    p = _ATOM_CACHE_DIR / (_group_cache_key(g) + ".json")
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if len(data) == len(g["sessions"]):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def extract_group_cached(backend, extract_cfg, g: dict) -> list:
    """组内抽取（带磁盘缓存）：V3 与 V5 共享，重跑/跨版本零成本。"""
    cached = _load_cached_atoms(g)
    if cached is not None:
        return cached
    import dataclasses
    extract_fast = dataclasses.replace(extract_cfg, thinking=False)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        extracted = list(ex.map(lambda s: extract_atoms(backend, extract_fast, s),
                                g["sessions"]))
    _ATOM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _ATOM_CACHE_DIR / (_group_cache_key(g) + ".json")
    p.write_text(json.dumps(extracted, ensure_ascii=False), encoding="utf-8")
    return extracted


def run(questions: list[dict], out_dir: str | Path, tag: str = "v3",
        model_cfg: ModelConfig | None = None, extract_cfg: ModelConfig | None = None,
        limit: int | None = None, workers: int = 3) -> dict:
    from systems.runner import run_pipeline
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = model_cfg or deepseek_config()
    extract_cfg = extract_cfg or model_cfg
    backend = OpenAICompatBackend()

    if limit:
        questions = questions[:limit]
    groups = group_by_haystack(questions)

    def build_group(g):
        store = MemoryStore(":memory:")
        atoms = AtomStore(store)
        extracted = extract_group_cached(backend, extract_cfg, g)
        for s, atom_dicts in zip(g["sessions"], extracted):
            atoms.apply_atoms(s, atom_dicts)
        index = DenseIndex()
        index.add(store.all_cards())
        return store, index

    def answer_one(ctx, q):
        store, index = ctx
        try:
            ans, meta = answer_question(store, index, backend, model_cfg,
                                        q["question"], q.get("question_date"))
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
    ap.add_argument("--out", default="runs/v3")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    load_env()
    questions = json.loads(Path(args.data).read_text(encoding="utf-8"))
    tag = "v3_sub%d" % args.subset if args.subset else "v3_full"
    if args.subset:
        questions = stratified_subset(questions, args.subset)
    run(questions, args.out, tag=tag, limit=args.limit or None)


if __name__ == "__main__":
    main()
