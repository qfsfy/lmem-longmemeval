"""LongMemEval 适配器：加载数据 → 驱动 harness → 输出官方格式 JSONL。

官方评测协议（GPT-4o judge）见：
https://github.com/xiaowu0162/LongMemEval  （evaluate_qa.py / print_qa_metrics.py）

本模块产出的 {question_id, hypothesis} JSONL 可直接交给官方脚本评分；
evals/judge.py 提供开发期快速判分（非官方口径）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..memory.schema import Session, Turn
from ..pipeline.harness import MemoryHarness


def load_questions(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.json")) + sorted(p.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"{p} 下没有找到 .json/.jsonl 数据文件（见目录内 README）")
        items: list[dict] = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            items.extend(json.loads(text) if f.suffix == ".jsonl" else json.loads(text))
        return items
    text = p.read_text(encoding="utf-8")
    return json.loads(text) if p.suffix == ".jsonl" else json.loads(text)


def _to_sessions(q: dict) -> list[Session]:
    ids = q.get("haystack_session_ids") or [str(i) for i in range(len(q["haystack_sessions"]))]
    dates = q.get("haystack_dates") or [""] * len(q["haystack_sessions"])
    sessions = []
    for sid, d, turns in zip(ids, dates, q["haystack_sessions"]):
        sessions.append(
            Session(
                session_id=str(sid),
                date=str(d)[:10],
                turns=[Turn(role=t.get("role", "user"), content=t.get("content", ""))
                       for t in turns],
            )
        )
    return sessions


def group_by_haystack(questions: list[dict]) -> list[dict]:
    """同一 haystack 的题目共享一次建库（显著省写入成本）。"""
    groups: dict[str, dict] = {}
    for q in questions:
        key = hashlib.md5(
            json.dumps(q.get("haystack_session_ids") or q["haystack_sessions"],
                       ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:12]
        g = groups.setdefault(key, {"sessions": _to_sessions(q), "questions": []})
        g["questions"].append(q)
    return list(groups.values())


def run_eval(
    harness: MemoryHarness,
    questions: list[dict],
    out_path: str | Path,
    limit: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """跑批：返回内部记录列表（供 judge/metrics），并写出官方格式 JSONL。"""
    if limit:
        questions = questions[:limit]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with out_path.open("w", encoding="utf-8") as f:
        for gi, g in enumerate(group_by_haystack(questions), start=1):
            harness.reset()
            write_stats = [harness.add_session(s) for s in g["sessions"]]
            if verbose:
                cards = sum(w["cards_added"] for w in write_stats)
                print(f"[组 {gi}/{len(group_by_haystack(questions))}] "
                      f"{len(g['sessions'])} 会话 → {cards} 张记忆卡")
            for q in g["questions"]:
                ans = harness.answer(q["question"], as_of=q.get("question_date"))
                f.write(json.dumps(
                    {"question_id": q["question_id"], "hypothesis": ans.answer},
                    ensure_ascii=False,
                ) + "\n")
                records.append({
                    "question_id": q["question_id"],
                    "question_type": q.get("question_type", "unknown"),
                    "answer": q.get("answer", ""),
                    "hypothesis": ans.answer,
                    "abstained": ans.abstained,
                })
    return records
