"""MemoryHarness —— 对外统一门面。

用法：
    harness = MemoryHarness()                 # auto：有 Key 走 GLM，无 Key 走 Mock
    harness.add_session(Session(...))         # 写入
    ans = harness.answer("我喜欢什么？")        # 读取+生成
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import HarnessConfig, load_config
from ..llm.client import BaseBackend, MockBackend, OpenAICompatBackend, make_backend
from ..memory.retriever import Retriever
from ..memory.schema import Session
from ..memory.store import MemoryStore
from ..memory.writer import ingest_session
from .gate import apply_gate
from .generator import AnswerObj, generate_answer
from .router import pick_model, should_upgrade


@dataclass
class HarnessAnswer:
    question: str
    answer: str
    citations: list[int] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    qtype: str = ""
    route: str = ""           # flash / flash->flagship
    model_used: str = ""
    gate_action: str = "pass"
    coverage: float = 0.0
    evidence: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    def meta_brief(self) -> dict:
        return {
            "qtype": self.qtype, "route": self.route, "model": self.model_used,
            "confidence": round(self.confidence, 2), "coverage": round(self.coverage, 2),
            "abstained": self.abstained, "gate": self.gate_action,
            "n_evidence": len(self.evidence), "timings_ms": self.timings_ms,
        }


class MemoryHarness:
    def __init__(self, config: Optional[HarnessConfig] = None) -> None:
        self.cfg = config or load_config()
        self.backend: BaseBackend = make_backend(self.cfg)
        self.store = MemoryStore(self.cfg.store_path if self.cfg.store_path else ":memory:")
        self.retriever = Retriever(self.store, self.backend, self.cfg.flash, self.cfg.retriever)

    # ------------------------------------------------------------- 写入

    def add_session(self, session: Session) -> dict:
        t0 = time.time()
        out = ingest_session(
            self.store, self.backend, self.cfg.flash, session,
            supersede_on_conflict=self.cfg.writer.supersede_on_conflict,
        )
        out["latency_ms"] = int((time.time() - t0) * 1000)
        return out

    def add_text(self, text: str, session_id: Optional[str] = None,
                 date: Optional[str] = None) -> dict:
        """便捷入口：把一段用户话术直接写入记忆（UI/CLI 演示用）。"""
        from ..memory.schema import Turn
        from ..textutils import today_str

        sid = session_id or f"adhoc-{int(time.time() * 1000)}"
        session = Session(
            session_id=sid, date=date or today_str(), turns=[Turn(role="user", content=text)]
        )
        return self.add_session(session)

    # ------------------------------------------------------------- 读取

    def answer(self, question: str, as_of: Optional[str] = None) -> HarnessAnswer:
        timings: dict = {}

        t0 = time.time()
        ev = self.retriever.retrieve(question, as_of=as_of)
        timings["retrieve_ms"] = int((time.time() - t0) * 1000)
        qtype = ev.analysis.qtype if ev.analysis else ""

        t0 = time.time()
        first_model = pick_model(self.cfg.router, qtype, self.cfg.flash, self.cfg.flagship)
        ans = self._gen(question, ev, as_of, qtype, first_model)
        route = "flagship" if first_model is self.cfg.flagship else "flash"

        if should_upgrade(self.cfg.router, ans, qtype):
            ans = self._gen(question, ev, as_of, qtype, self.cfg.flagship)
            route = "flash->flagship"
        timings["generate_ms"] = int((time.time() - t0) * 1000)

        final, action = apply_gate(ans, ev.coverage, self.cfg.gate)

        return HarnessAnswer(
            question=question, answer=final.answer, citations=final.citations,
            confidence=final.confidence, abstained=final.abstained, qtype=qtype,
            route=route, model_used=first_model.name if route != "flash->flagship"
            else self.cfg.flagship.name, gate_action=action, coverage=ev.coverage,
            evidence=ev.briefs(), timings_ms=timings,
        )

    def _gen(self, question, ev, as_of, qtype, model_cfg) -> AnswerObj:
        return generate_answer(self.backend, model_cfg, question, ev.cards, as_of, qtype)

    # ------------------------------------------------------------- 其他

    def stats(self) -> dict:
        s = self.store.stats()
        s["backend"] = self.cfg.effective_backend()
        s["models"] = {"flash": self.cfg.flash.name, "flagship": self.cfg.flagship.name}
        s["router"] = self.cfg.router.strategy
        return s

    def reset(self) -> None:
        for t in ("cards", "sessions", "meta"):
            self.store.conn.execute(f"DELETE FROM {t}")
        self.store.conn.commit()
        self.store._dirty = True
