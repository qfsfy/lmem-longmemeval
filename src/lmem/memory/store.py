"""SQLite 持久化 + 进程内 BM25 索引。

规模假设：单用户记忆库为 10^3~10^5 张卡片，BM25 全量内存索引足够；
稠密向量检索在 roadmap（research/06 §3）中作为第二路召回接入，接口已预留。
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from pathlib import Path

from .schema import MemoryCard
from ..textutils import tokenize

_DDL = """
CREATE TABLE IF NOT EXISTS cards(
  id TEXT PRIMARY KEY, kind TEXT, content TEXT, subject TEXT, predicate TEXT, value TEXT,
  valid_from TEXT, valid_to TEXT, observed_at TEXT, session_id TEXT,
  turn_start INTEGER, turn_end INTEGER, confidence REAL, version INTEGER, superseded_by TEXT
);
CREATE TABLE IF NOT EXISTS sessions(
  session_id TEXT PRIMARY KEY, date TEXT, n_turns INTEGER
);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""


class _BM25:
    """标准 BM25（k1=1.5, b=0.75），支持 allowed id 过滤。"""

    def __init__(self) -> None:
        self._docs: dict[str, Counter] = {}
        self._len: dict[str, int] = {}
        self._df: Counter = Counter()
        self._avgdl = 0.0

    def rebuild(self, docs: list[tuple[str, str]]) -> None:
        self._docs, self._len, self._df = {}, {}, Counter()
        for cid, text in docs:
            toks = tokenize(text)
            cnt = Counter(toks)
            self._docs[cid] = cnt
            self._len[cid] = len(toks)
            for t in set(toks):
                self._df[t] += 1
        self._avgdl = (sum(self._len.values()) / len(self._len)) if self._len else 0.0

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if not self._docs:
            return []
        q = Counter(tokenize(query))
        n = len(self._docs)
        scores: list[tuple[str, float]] = []
        for cid, cnt in self._docs.items():
            if allowed is not None and cid not in allowed:
                continue
            dl = self._len[cid] or 1
            s = 0.0
            for t, qtf in q.items():
                if qtf == 0 or t not in cnt:
                    continue
                df = self._df.get(t, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = cnt[t]
                s += idf * (tf * 2.2) / (tf + 1.2 * (0.75 * (dl / (self._avgdl or 1)) + 0.25))
            if s > 0:
                scores.append((cid, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：并行评测时建库在主线程、答题在工作线程，
        # 单个 store 永远不会被两个线程同时使用
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(_DDL)
        self._index = _BM25()
        self._dirty = True

    # ---------------------------------------------------------- 写入

    def add_card(self, card: MemoryCard) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", card.to_row()
        )
        self.conn.commit()
        self._dirty = True
        return card.id

    def add_session(self, session_id: str, date: str, n_turns: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?)", (session_id, date, n_turns)
        )
        self.conn.commit()

    def supersede(self, old_id: str, new_id: str, date: str) -> None:
        """旧卡失效：valid_to 置为新卡观测日期（保留历史，支持 as-of 查询）。"""
        self.conn.execute(
            "UPDATE cards SET valid_to=?, superseded_by=?, version=version WHERE id=?",
            (date, new_id, old_id),
        )
        self.conn.commit()
        self._dirty = True

    # ---------------------------------------------------------- 读取

    def get(self, card_id: str) -> MemoryCard | None:
        r = self.conn.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
        return MemoryCard.from_row(r) if r else None

    def all_cards(self, include_invalid: bool = True) -> list[MemoryCard]:
        rows = self.conn.execute("SELECT * FROM cards").fetchall()
        cards = [MemoryCard.from_row(r) for r in rows]
        if not include_invalid:
            cards = [c for c in cards if c.valid_to is None]
        return cards

    def active_at(self, date: str | None) -> list[MemoryCard]:
        return [c for c in self.all_cards() if c.active_at(date)]

    def sessions(self) -> list[tuple[str, str, int]]:
        return self.conn.execute("SELECT session_id, date, n_turns FROM sessions").fetchall()

    def stats(self) -> dict:
        cards = self.all_cards()
        by_kind: dict[str, int] = {}
        for c in cards:
            by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
        return {
            "sessions": len(self.sessions()),
            "cards": len(cards),
            "by_kind": by_kind,
            "active": sum(1 for c in cards if c.valid_to is None),
        }

    # ---------------------------------------------------------- 检索

    def _ensure_index(self) -> None:
        if self._dirty:
            self._index.rebuild([(c.id, c.content) for c in self.all_cards()])
            self._dirty = False

    def search(
        self, query: str, k: int, date: str | None = None, kinds: list[str] | None = None
    ) -> list[tuple[MemoryCard, float]]:
        """BM25 检索；date 给定时只召回当时仍有效的卡片。"""
        self._ensure_index()
        pool = self.active_at(date) if date else self.all_cards()
        if kinds:
            pool = [c for c in pool if c.kind in kinds]
        allowed = {c.id for c in pool}
        hits = self._index.search(query, k, allowed)
        by_id = {c.id: c for c in pool}
        return [(by_id[cid], s) for cid, s in hits if cid in by_id]

    def close(self) -> None:
        self.conn.close()
