"""开发期快速判分（非官方口径）。

正式对外报告必须用官方 evaluate_qa.py（GPT-4o judge），
本模块用于迭代时的快速反馈与消融对比；judge 模型走 config.judge。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import ModelConfig
from ..llm.client import BaseBackend

JUDGE_SYSTEM = """你是严格的问答判卷人。给定问题、标准答案与候选答案，
判断候选答案是否与标准答案在事实上一致（语义等价即可，不要求逐字匹配）。
注意：候选答案明确表示"不知道/无法回答"时应判 incorrect，除非标准答案本身就是拒答。"""


class JudgeOut(BaseModel):
    verdict: str = Field(description="correct 或 incorrect")
    reason: str = ""


def judge_one(backend: BaseBackend, model_cfg: ModelConfig, question: str,
              answer: str, hypothesis: str) -> JudgeOut:
    comp = backend.complete(
        task="judge", model_cfg=model_cfg, system=JUDGE_SYSTEM,
        user=f"问题：{question}\n标准答案：{answer}\n候选答案：{hypothesis}",
        schema=JudgeOut,
        extra={"question": question, "answer": answer, "hypothesis": hypothesis},
    )
    if comp.data is None:
        return JudgeOut(verdict="incorrect", reason="judge 输出解析失败")
    return comp.data
