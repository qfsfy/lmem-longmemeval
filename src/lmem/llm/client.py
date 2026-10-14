"""LLM 后端抽象。

- OpenAICompatBackend：任何 OpenAI 兼容端点（智谱/Z.ai/OpenAI/本地 vLLM）；
- MockBackend：零依赖离线演示——规则化抽取 + 启发式查询分析 + 模板生成，
  让未配 Key 的用户也能完整体验 harness 全流程（含 UI）。

complete() 按 task 区分用途：extract / conflict / analyze_query / generate / judge。
Mock 后端依赖 extra 里的结构化数据，不解析自然语言提示词。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

from ..config import ModelConfig
from ..textutils import tokenize


@dataclass
class Completion:
    text: str = ""
    data: Optional[BaseModel] = None
    usage: dict = field(default_factory=dict)
    model: str = ""
    latency_ms: int = 0


class BaseBackend:
    def complete(
        self,
        *,
        task: str,
        model_cfg: ModelConfig,
        system: str = "",
        user: str = "",
        schema: Optional[Type[BaseModel]] = None,
        extra: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Completion:
        raise NotImplementedError


class OpenAICompatBackend(BaseBackend):
    """OpenAI 兼容 Chat Completions 客户端（含 JSON 模式 + 一次解析重试）。"""

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def _client(self, model_cfg: ModelConfig):
        key = (model_cfg.base_url, model_cfg.api_key_env)
        if key not in self._clients:
            from openai import OpenAI  # 延迟导入：Mock 模式无需安装

            self._clients[key] = OpenAI(
                base_url=model_cfg.base_url, api_key=model_cfg.api_key() or "EMPTY"
            )
        return self._clients[key]

    @staticmethod
    def _extra_body(model_cfg: ModelConfig) -> dict:
        body = dict(model_cfg.extra_body)
        if model_cfg.thinking is not None:
            body.setdefault("thinking", {"type": "enabled" if model_cfg.thinking else "disabled"})
        return body

    def complete(self, *, task, model_cfg, system="", user="", schema=None,
                 extra=None, temperature=None, max_tokens=None) -> Completion:
        client = self._client(model_cfg)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if schema is not None:
            schema_json = json.dumps(_slim_schema(schema), ensure_ascii=False)
            messages[0]["content"] += (
                "\n\n输出要求：只输出一个符合以下 JSON Schema 的 JSON 对象，"
                f"不要输出任何其他文字。\n{schema_json}"
            )
        kwargs: dict[str, Any] = dict(
            model=model_cfg.name,
            messages=messages,
            temperature=model_cfg.temperature if temperature is None else temperature,
            max_tokens=model_cfg.max_tokens if max_tokens is None else max_tokens,
        )
        eb = self._extra_body(model_cfg)
        if eb:
            kwargs["extra_body"] = eb
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.time()
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }

        data = None
        if schema is not None:
            data = _try_parse(schema, text)
            if data is None:  # 一次重试：把解析错误反馈给模型
                retry_user = (
                    f"{user}\n\n你上一次的输出无法按 Schema 解析，请严格重新输出 JSON：\n{text[:2000]}"
                )
                kwargs["messages"] = messages + [{"role": "user", "content": retry_user}]
                resp = client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                data = _try_parse(schema, text)
        return Completion(
            text=text,
            data=data,
            usage=usage,
            model=model_cfg.name,
            latency_ms=int((time.time() - t0) * 1000),
        )


def make_backend(cfg) -> BaseBackend:
    mode = cfg.effective_backend() if hasattr(cfg, "effective_backend") else str(cfg)
    return MockBackend() if mode == "mock" else OpenAICompatBackend()


class MockBackend(BaseBackend):
    """离线规则后端：演示模式下撑起全部 task。"""

    def complete(self, *, task, model_cfg, system="", user="", schema=None,
                 extra=None, temperature=None, max_tokens=None) -> Completion:
        extra = extra or {}
        if task == "extract":
            data = _mock_extract(schema, extra)
        elif task == "analyze_query":
            data = _mock_analyze(schema, extra.get("question", ""))
        elif task == "generate":
            data = _mock_generate(schema, extra)
        elif task == "judge":
            data = _mock_judge(schema, extra)
        else:
            data = None
        text = data.model_dump_json() if data is not None else ""
        return Completion(text=text, data=data, model="mock", usage={}, latency_ms=0)


# ---------------------------------------------------------------- 工具函数

def _slim_schema(schema: Type[BaseModel]) -> dict:
    """pydantic -> 精简 JSON Schema（去掉冗长字段说明之外的内容）。"""
    s = schema.model_json_schema()
    s.pop("$defs", None)
    s.pop("definitions", None)
    return s


def _try_parse(schema: Type[BaseModel], text: str) -> Optional[BaseModel]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return schema.model_validate_json(m.group(0))
    except ValidationError:
        return None


def _mock_extract(schema, extra: dict) -> BaseModel:
    """规则化事实抽取：覆盖演示常用句式（中英）。"""
    turns = extra.get("turns", [])
    date = extra.get("date", "")
    facts: list[dict] = []
    pats = [
        (r"(?:我叫|我的名字是|我叫作)\s*([\u4e00-\u9fa5A-Za-z]{1,12})", "name", "名字"),
        (r"(?:我(?:现在|以前|一直|最近)?(?:特别|更|最)?(?:喜欢|爱)|我喜欢上了)\s*"
         r"(?:穿|吃|用|听|玩|看)?\s*"
         r"([\u4e00-\u9fa5A-Za-z0-9 ]{1,20}?)(?:[，。,.!！]|$)", "likes", "喜好"),
        (r"(?:我住在|我搬到了?|搬到|移居(?:到|在)?|定居(?:在|于)?|搬去|I (?:live in|moved to))\s*"
         r"([\u4e00-\u9fa5A-Za-z]{1,20}?)(?:[，。,.!！]|$)", "lives_in", "居住地"),
        (r"(?:我是一名|我是|我担任|我在做|我做|现在做|如今做|I (?:am|work as))\s*(?:一名)?"
         r"([\u4e00-\u9fa5A-Za-z]{1,12}?)(?:[，。,.!！]|$)", "job", "职业"),
        (r"(?:我(?:的生日)?(?:是|在)\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})|my birthday is (\S+))",
         "birthday", "生日"),
    ]
    cleanups = {"job": lambda v: re.sub(r"(的)?(新)?工作$", "", v)}

    def _clean(v: str, pred: str) -> str:
        v = re.sub(r"[了的啦哦]+$", "", v).strip()
        if pred in cleanups:
            v = cleanups[pred](v).strip()
        return v

    for t in turns:
        if t.get("role") != "user":
            continue
        for pat, pred, label in pats:
            for m in re.finditer(pat, t.get("content", "")):
                value = next((g for g in m.groups() if g), "").strip()
                value = _clean(value, pred)
                if value:
                    facts.append({
                        "subject": "user", "predicate": pred, "value": value,
                        "content": f"用户{label}：{value}", "confidence": 0.9,
                    })
    facts = [f for f in facts if f["confidence"] >= 0.5]
    return schema.model_validate({"cards": facts, "date": date})


def _mock_analyze(schema, question: str) -> BaseModel:
    q = question.lower()
    time_hit = re.search(
        r"(昨天|今天|去年|今年|上周|上个月|现在|最近|\d{4}年|\d{1,2}月|\d{1,2}日|yesterday|last \w+|now)", q
    )
    if time_hit:
        qtype = "temporal_reasoning"
    elif re.search(r"(以前|之前|原来|曾经|过去|used to)", q):
        qtype = "knowledge_update"
    elif re.search(r"(哪些|几次|每次|分别|一共|all|how many|every)", q):
        qtype = "multi_session"
    elif re.search(r"(喜欢|爱好|偏好|讨厌|like|prefer|favorite)", q):
        qtype = "single_session_preference"
    elif re.search(r"我|i ", q):
        qtype = "single_session_user"
    else:
        qtype = "single_session_assistant"
    kws = list(dict.fromkeys(tokenize(question)))
    return schema.model_validate({
        "qtype": qtype,
        "entities": [],
        "time_expressions": [time_hit.group(0)] if time_hit else [],
        "subqueries": [],
        "keywords": kws[:24],
    })


def _mock_generate(schema, extra: dict) -> BaseModel:
    evidence = extra.get("evidence", [])
    abstain_hint = extra.get("abstain_hint", False)
    if not evidence or abstain_hint:
        return schema.model_validate({
            "answer": "（Mock 演示）记忆中没有足够的证据回答这个问题，我选择拒答。",
            "citations": [], "confidence": 0.9, "abstained": True,
        })
    cites = [e["cite"] for e in evidence[:2]]
    lines = [f"[cite:{e['cite']}] {e['content']}（{e.get('valid_from', '')}）"
             for e in evidence[:2]]
    answer = (
        "（Mock 演示）根据记忆库中的证据：\n" + "\n".join(lines)
    )
    return schema.model_validate({
        "answer": answer, "citations": cites, "confidence": 0.8, "abstained": False,
    })


def _mock_judge(schema, extra: dict) -> BaseModel:
    # 演示判卷：答案与标准答案有任意重叠 token 即算 correct
    hyp = set(tokenize(extra.get("hypothesis", "")))
    gold = set(tokenize(extra.get("answer", "")))
    overlap = hyp & gold
    verdict = "correct" if overlap and len(overlap) >= max(1, len(gold) * 0.3) else "incorrect"
    return schema.model_validate({"verdict": verdict, "reason": "mock 重叠率判定"})
