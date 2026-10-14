"""拒答门控：反幻觉出口（对齐 LongMemEval abstention 题型）。"""

from __future__ import annotations

from ..config import GateConfig
from .generator import AnswerObj

ABSTAIN_TEMPLATE = (
    "根据现有记忆，我无法确定地回答这个问题。"
    "如果你能提供更多背景，或确认相关信息，我可以再试一次。"
)


def apply_gate(answer: AnswerObj, coverage: float, cfg: GateConfig) -> tuple[AnswerObj, str]:
    """返回 (最终答案, 门控动作)。

    动作：pass=照常输出；abstain_coverage=证据不足拒答；abstain_confidence=低置信且无证据拒答。
    """
    if not answer.abstained and coverage < cfg.coverage_min:
        gated = AnswerObj(
            answer=ABSTAIN_TEMPLATE, citations=[], confidence=answer.confidence, abstained=True
        )
        return gated, "abstain_coverage"
    if (not answer.abstained and not answer.citations
            and answer.confidence < cfg.confidence_min):
        gated = AnswerObj(
            answer=ABSTAIN_TEMPLATE, citations=[], confidence=answer.confidence, abstained=True
        )
        return gated, "abstain_confidence"
    return answer, "pass"
