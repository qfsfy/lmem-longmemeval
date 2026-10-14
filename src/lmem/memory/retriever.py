"""读取管线：查询分析 → 多路召回 → 证据打包 + 覆盖率评估。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from ..config import ModelConfig
from ..llm.client import BaseBackend
from ..textutils import tokenize
from .schema import MemoryCard
from .store import MemoryStore


class QueryAnalysis(BaseModel):
    qtype: str = Field(default="single_session_user", description="题型分类")
    entities: list[str] = Field(default_factory=list)
    time_expressions: list[str] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


ANALYZE_SYSTEM = """你是记忆检索的查询分析器。对用户问题输出 JSON：
- qtype：single_session_user / single_session_assistant / single_session_preference /
  multi_session / temporal_reasoning / knowledge_update 之一；
- entities：提到的实体名；time_expressions：时间表达（原样）；
- subqueries：若需要多跳/跨会话，给出拆解后的子查询（至多 3 条）；
- keywords：用于 BM25 检索的关键词列表（8~24 个，中英混合时都给）。"""


@dataclass
class Evidence:
    cards: list[MemoryCard] = field(default_factory=list)
    coverage: float = 0.0
    analysis: Optional[QueryAnalysis] = None

    def briefs(self) -> list[dict]:
        return [c.brief(cite=i + 1) for i, c in enumerate(self.cards)]


class Retriever:
    def __init__(self, store: MemoryStore, backend: BaseBackend, flash_cfg: ModelConfig,
                 retriever_cfg) -> None:
        self.store = store
        self.backend = backend
        self.flash_cfg = flash_cfg
        self.cfg = retriever_cfg

    # -- 阶段 1：查询分析 ------------------------------------------------
    def analyze(self, question: str) -> QueryAnalysis:
        comp = self.backend.complete(
            task="analyze_query", model_cfg=self.flash_cfg,
            system=ANALYZE_SYSTEM, user=question,
            schema=QueryAnalysis, extra={"question": question},
        )
        if comp.data is None:
            kws = list(dict.fromkeys(tokenize(question)))
            return QueryAnalysis(keywords=kws)
        return comp.data

    # -- 阶段 2+3：召回 + 打包 -------------------------------------------
    def retrieve(self, question: str, as_of: Optional[str] = None) -> Evidence:
        analysis = self.analyze(question)
        date = as_of if self.cfg.as_of_filter else None
        if analysis.qtype == "knowledge_update":
            # 历史/更新类问题需要看到"已失效"的旧记忆（时间旅行查询）
            date = None

        query = question + " " + " ".join(analysis.subqueries)
        hits = self.store.search(query, k=self.cfg.bm25_top_k, date=date)
        hit_cards = [c for c, _ in hits]

        # 画像直取：身份/偏好问题直接补齐有效画像卡（不计入覆盖率）
        profiles = [c for c in self.store.active_at(date) if c.kind == "profile"]
        profiles.sort(key=lambda c: c.observed_at, reverse=True)
        existing = {c.id for c in hit_cards}
        padding = [c for c in profiles[: self.cfg.profile_max] if c.id not in existing]

        cards = hit_cards + padding
        coverage = self._coverage(hit_cards)
        return Evidence(cards=cards, coverage=coverage, analysis=analysis)

    # -- 阶段 4：覆盖率评估 -----------------------------------------------
    # 脚手架启发式：以 BM25 是否有相关命中为信号（无命中 → 0 → 门控拒答）。
    # roadmap：升级为 Flash 结构化证据充分性评估（research/06 §3）。
    @staticmethod
    def _coverage(hit_cards: list[MemoryCard]) -> float:
        if not hit_cards:
            return 0.0
        return min(1.0, 0.45 + 0.05 * len(hit_cards))
