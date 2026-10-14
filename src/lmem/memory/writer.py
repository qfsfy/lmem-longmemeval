"""写入管线：会话 → 记忆卡（抽取 + 冲突失效）。"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..config import ModelConfig
from ..llm.client import BaseBackend
from .schema import MemoryCard, Session
from .store import MemoryStore


class ExtractOut(BaseModel):
    cards: list[dict] = Field(default_factory=list)
    date: str = ""


EXTRACT_SYSTEM = """你是记忆写入引擎。从给定的多轮对话中抽取值得长期保存的记忆卡。
规则：
1. 只抽取用户的稳定事实（身份/居住地/职业/生日）、偏好（喜欢/讨厌/偏好）、重要事件；
2. content 用一句完整中文陈述（含主语）；predicate 从 name/likes/dislikes/lives_in/job/birthday/other 中选；
3. confidence 为 0~1，推断而非明说的给 0.5~0.7；
4. 宁缺毋滥：寒暄、闲聊、无信息量的轮次不要抽取。"""


def extract_cards(backend: BaseBackend, model_cfg: ModelConfig, session: Session) -> list[MemoryCard]:
    """会话 → 记忆卡（fact/profile 不区分来源，均由同一抽取器产出）。"""
    dialogue = "\n".join(f"[{t.role}] {t.content}" for t in session.turns)
    comp = backend.complete(
        task="extract",
        model_cfg=model_cfg,
        system=EXTRACT_SYSTEM,
        user=f"会话日期：{session.date}\n对话内容：\n{dialogue}",
        schema=ExtractOut,
        extra={"turns": [{"role": t.role, "content": t.content} for t in session.turns],
               "date": session.date},
    )
    cards: list[MemoryCard] = []
    if comp.data is None:
        return cards
    for d in comp.data.cards:
        kind = "profile" if d.get("predicate") in ("name", "job", "birthday", "lives_in") else "fact"
        cards.append(
            MemoryCard(
                kind=kind,
                content=str(d.get("content", "")).strip(),
                subject=str(d.get("subject", "user")),
                predicate=str(d.get("predicate", "other")),
                value=str(d.get("value", "")).strip(),
                valid_from=session.date,
                observed_at=session.date,
                session_id=session.session_id,
                turn_start=0,
                turn_end=max(len(session.turns) - 1, 0),
                confidence=float(d.get("confidence", 0.8)),
            )
        )
    return [c for c in cards if c.content]


def apply_conflicts(
    store: MemoryStore,
    new_cards: list[MemoryCard],
    date: str,
    supersede_on_conflict: bool = True,
) -> list[str]:
    """冲突检测与失效：同 subject+predicate 且 value 不同 → 旧卡置为失效。

    设计说明（对应 research/06 §3）：写入期用规则判定"值不同即冲突"，
    语义级冲突仲裁（LLM 判断）接入点在此函数内，roadmap 中升级为 Flash 判定 + 5.3 仲裁。
    返回被失效的旧卡 id 列表。
    """
    superseded: list[str] = []
    if not supersede_on_conflict:
        return superseded
    for new in new_cards:
        if not new.predicate or new.predicate == "other":
            continue
        for old in store.active_at(date):
            if old.id == new.id or old.kind not in ("fact", "profile"):
                continue
            if old.subject == new.subject and old.predicate == new.predicate \
                    and old.value and new.value and old.value != new.value:
                store.supersede(old.id, new.id, date)
                superseded.append(old.id)
    return superseded


def ingest_session(
    store: MemoryStore,
    backend: BaseBackend,
    model_cfg: ModelConfig,
    session: Session,
    supersede_on_conflict: bool = True,
) -> dict:
    """单会话入库：抽取 → 冲突失效 → 落库。幂等：同 session 重复入库先清空旧卡。"""
    for old in store.all_cards():
        if old.session_id == session.session_id:
            store.conn.execute("DELETE FROM cards WHERE id=?", (old.id,))
    store.add_session(session.session_id, session.date, len(session.turns))
    cards = extract_cards(backend, model_cfg, session)
    superseded = apply_conflicts(store, cards, session.date, supersede_on_conflict)
    for c in cards:
        store.add_card(c)
    return {"cards_added": len(cards), "superseded": len(superseded),
            "superseded_ids": superseded}
