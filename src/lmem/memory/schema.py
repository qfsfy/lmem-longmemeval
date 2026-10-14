"""记忆数据模式。

核心约定（对应 research/06 设计）：
- 每张记忆卡带双时间轴：valid_from/valid_to（事实有效期）+ observed_at（系统得知时间）；
- 冲突不物理删除：旧卡 valid_to 被置为新卡的 observed_at，并记录 superseded_by，
  从而支持"时间旅行"查询（as-of）与时间推理题型；
- 来源可溯：session_id + turn 区间，引用即评测。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Session:
    session_id: str
    date: str  # YYYY-MM-DD
    turns: list[Turn] = field(default_factory=list)


@dataclass
class MemoryCard:
    kind: str  # fact | profile | summary
    content: str
    subject: str = "user"
    predicate: str = ""
    value: str = ""
    valid_from: str = ""
    valid_to: str | None = None
    observed_at: str = ""
    session_id: str = ""
    turn_start: int = 0
    turn_end: int = 0
    confidence: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    version: int = 1
    superseded_by: str | None = None

    def active_at(self, date: str | None) -> bool:
        """date 为 None 表示不过滤（取最新状态）。

        边界语义：valid_to 当日起旧卡不再生效（新卡当日已生效）。
        """
        if date is None:
            return self.valid_to is None
        if self.valid_from and date < self.valid_from:
            return False
        return self.valid_to is None or date < self.valid_to

    def to_row(self) -> tuple:
        return (
            self.id, self.kind, self.content, self.subject, self.predicate, self.value,
            self.valid_from, self.valid_to, self.observed_at, self.session_id,
            self.turn_start, self.turn_end, self.confidence, self.version, self.superseded_by,
        )

    @classmethod
    def from_row(cls, r) -> "MemoryCard":
        return cls(
            id=r[0], kind=r[1], content=r[2], subject=r[3], predicate=r[4], value=r[5],
            valid_from=r[6] or "", valid_to=r[7], observed_at=r[8] or "",
            session_id=r[9] or "", turn_start=r[10] or 0, turn_end=r[11] or 0,
            confidence=r[12] if r[12] is not None else 1.0,
            version=r[13] or 1, superseded_by=r[14],
        )

    def brief(self, cite: int | None = None) -> dict:
        d = {
            "id": self.id, "kind": self.kind, "content": self.content,
            "valid_from": self.valid_from, "valid_to": self.valid_to,
            "session_id": self.session_id, "confidence": round(self.confidence, 2),
        }
        if cite is not None:
            d["cite"] = cite
        return d
