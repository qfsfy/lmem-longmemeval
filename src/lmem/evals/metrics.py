"""指标汇总：分题型准确率、拒答混淆、成本统计。"""

from __future__ import annotations


def summarize(records: list[dict], judged: list[dict] | None = None) -> dict:
    """judged: [{"question_type","verdict","abstained"}]，与 records 对齐。"""
    src = judged or [
        {"question_type": r["question_type"],
         "verdict": "correct" if r["answer"].strip() and r["answer"] == r["hypothesis"] else "incorrect",
         "abstained": r.get("abstained", False)}
        for r in records
    ]
    by_type: dict[str, dict] = {}
    for r in src:
        t = r["question_type"]
        d = by_type.setdefault(t, {"n": 0, "correct": 0, "abstained": 0})
        d["n"] += 1
        d["correct"] += r["verdict"] == "correct"
        d["abstained"] += bool(r.get("abstained"))
    out = {}
    for t, d in sorted(by_type.items()):
        out[t] = {
            "n": d["n"],
            "accuracy": round(d["correct"] / d["n"], 4) if d["n"] else 0.0,
            "abstain_rate": round(d["abstained"] / d["n"], 4) if d["n"] else 0.0,
        }
    total = len(src)
    correct = sum(1 for r in src if r["verdict"] == "correct")
    return {"overall_accuracy": round(correct / total, 4) if total else 0.0,
            "n": total, "by_type": out}


def format_report(summary: dict) -> str:
    lines = [f"总体准确率：{summary['overall_accuracy']:.2%}（n={summary['n']}）", ""]
    lines.append(f"{'题型':<32}{'n':>5}{'准确率':>10}{'拒答率':>10}")
    lines.append("-" * 60)
    for t, d in summary["by_type"].items():
        lines.append(f"{t:<32}{d['n']:>5}{d['accuracy']:>10.2%}{d['abstain_rate']:>10.2%}")
    return "\n".join(lines)
