"""答案生成：证据打包 + 结构化输出（引用 / 置信度 / 弃权标记）。"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional

from ..config import ModelConfig
from ..llm.client import BaseBackend
from ..memory.schema import MemoryCard

GEN_SYSTEM = """你是长期记忆助手的回答引擎。回答必须遵守三条原则：
1. 只依据给出的记忆证据回答，并在答案中用 [cite:编号] 标注所引用的证据；
2. 证据不足或互相矛盾时，宁可弃权：设 abstained=true，answer 写明无法确定的原因；
3. 涉及时间的问题必须基于证据里的日期推算，不要凭空猜测。"""


class AnswerObj(BaseModel):
    answer: str = ""
    citations: list[int] = Field(default_factory=list)
    confidence: float = 0.5
    abstained: bool = False


def build_context(cards: list[MemoryCard]) -> str:
    lines = []
    for i, c in enumerate(cards, start=1):
        valid = c.valid_from + ("~仍有效" if not c.valid_to else f"~{c.valid_to}（已失效）")
        lines.append(f"[cite:{i}]（{c.session_id} {valid}）{c.content}")
    return "\n".join(lines) if lines else "（无证据）"


def generate_answer(
    backend: BaseBackend,
    model_cfg: ModelConfig,
    question: str,
    cards: list[MemoryCard],
    as_of: Optional[str] = None,
    qtype: str = "",
) -> AnswerObj:
    evidence_briefs = [c.brief(cite=i + 1) for i, c in enumerate(cards)]
    context = build_context(cards)
    user = (
        f"问题日期：{as_of or '未知'}\n查询类型：{qtype or '未知'}\n"
        f"问题：{question}\n\n记忆证据：\n{context}"
    )
    comp = backend.complete(
        task="generate",
        model_cfg=model_cfg,
        system=GEN_SYSTEM,
        user=user,
        schema=AnswerObj,
        extra={"evidence": evidence_briefs, "question": question, "abstain_hint": not cards},
    )
    if comp.data is None:
        return AnswerObj(answer="生成失败：模型输出无法解析。", confidence=0.0, abstained=True)
    return comp.data
