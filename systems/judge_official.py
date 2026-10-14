"""LongMemEval 官方口径判分（开发期快速复现）。

grader prompt 逐字来自官方脚本（5 种题型模板 + abstention 模板）：
    https://github.com/xiaowu0162/LongMemEval  →  src/evaluation/evaluate_qa.py
官方口径 judge = GPT-4o；本模块允许替换 judge 端点（任何 OpenAI 兼容 API），
开发期用 DeepSeek 判分仅用于迭代相对比较，最终 Benchmark 得分以项目组
统一复跑（GPT-4o judge）为准 —— 见手册附录 C.3。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from systems.v1_lexical import deepseek_config, load_env  # 复用端点配置加载

ANSCHECK_TEMPLATES: dict[str, str] = {
    "default": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. In addition, "
        "do not penalize off-by-one errors for the number of days. If the question asks for the "
        "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
        "predicting 19 days when the answer is 18), the model's response is still correct. "
        "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, the "
        "response should be considered as correct as long as the updated answer is the required "
        "answer."
        "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. "
        "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
        "The response is correct as long as it recalls and utilizes the user's personal "
        "information correctly."
        "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is "
        "given but the asked information is not."
        "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}


def grader_prompt(qtype: str, question: str, answer: str, response: str,
                  is_abs: bool) -> str:
    if is_abs:
        return ANSCHECK_TEMPLATES["abstention"].format(question, answer, response)
    if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        return ANSCHECK_TEMPLATES["default"].format(question, answer, response)
    return ANSCHECK_TEMPLATES[qtype].format(question, answer, response)


def judge_file(hyp_path: str | Path, ref_path: str | Path, out_path: str | Path | None = None,
               base_url: str | None = None, api_key: str | None = None,
               model: str | None = None) -> dict:
    import os
    # 默认判分端点优先级：显式 JUDGE_* > 智谱 coding plan > DeepSeek
    if os.environ.get("JUDGE_BASE_URL"):
        base_url = os.environ["JUDGE_BASE_URL"]
    elif os.environ.get("ZAI_CODING_KEY"):
        base_url = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")
    else:
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    if os.environ.get("JUDGE_API_KEY"):
        api_key = os.environ["JUDGE_API_KEY"]
    elif os.environ.get("ZAI_CODING_KEY"):
        api_key = os.environ["ZAI_CODING_KEY"]
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if os.environ.get("JUDGE_MODEL"):
        model = os.environ["JUDGE_MODEL"]
    elif os.environ.get("ZAI_CODING_KEY"):
        model = os.environ.get("ZHIPU_MODEL", "glm-5.3-flash")
    else:
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)

    hyps = [json.loads(l) for l in Path(hyp_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    refs = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    qid2ref = {r["question_id"]: r for r in refs}

    out_path = Path(out_path or str(hyp_path) + f".judged-{model}")
    acc_by_type: dict[str, list[int]] = defaultdict(list)
    n_ok = n_all = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for i, h in enumerate(hyps, 1):
            ref = qid2ref.get(h["question_id"])
            if ref is None:
                continue
            is_abs = h["question_id"].endswith("_abs")
            prompt = grader_prompt(ref["question_type"], ref["question"], ref["answer"],
                                   h["hypothesis"], is_abs)
            label = None
            for attempt in range(5):
                try:
                    r = client.chat.completions.create(
                        model=model, temperature=0, max_tokens=10,
                        messages=[{"role": "user", "content": prompt}])
                    label = "yes" in (r.choices[0].message.content or "").strip().lower()
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"retry {attempt + 1} ({e}); sleep {wait}s", flush=True)
                    time.sleep(wait)
            if label is None:
                label = False
            qtype = ref["question_type"] + ("-abs" if is_abs else "")
            acc_by_type[qtype].append(1 if label else 0)
            n_ok += 1 if label else 0
            n_all += 1
            f.write(json.dumps({**h, "autoeval_label": {"model": model, "label": label}},
                               ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"judged {i}/{len(hyps)} acc={n_ok / max(n_all, 1):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    overall = n_ok / max(n_all, 1)
    print(f"\n== Overall Accuracy: {overall:.4f}  (n={n_all}, judge={model}) ==")
    for t in sorted(acc_by_type):
        v = acc_by_type[t]
        print(f"  {t:38s} {sum(v) / len(v):.4f} ({len(v)})")
    result = {
        "judge_model": model, "n": n_all, "overall_accuracy": round(overall, 4),
        "by_type": {t: {"acc": round(sum(v) / len(v), 4), "n": len(v)}
                    for t, v in sorted(acc_by_type.items())},
    }
    Path(str(out_path) + ".summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hyp_file")
    ap.add_argument("--ref", default="data/longmemeval/longmemeval_oracle.json")
    args = ap.parse_args()
    load_env()
    judge_file(args.hyp_file, args.ref)


if __name__ == "__main__":
    main()
