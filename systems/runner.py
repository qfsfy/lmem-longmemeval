"""通用并行评测流水线。

所有版本共享的执行骨架：建库（每组一次）→ 任务展开 → 线程池并行答题（打到
llama-server 的多个 slot 上，GPU 批处理）→ 按原顺序写出 JSONL 与 records。

llama-server 端以 -np 4（自动）+ 统一 KV 运行，3 个并发客户端请求可以充分
利用 GPU 的 prompt 批处理；workers>服务端并发数时请求在服务端排队，无害。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

_print_lock = threading.Lock()


def run_pipeline(
    groups: list[dict],
    build_group: Callable,
    answer_one: Callable,
    out_dir: str | Path,
    tag: str,
    records_name: str = "records",
    workers: int = 3,
) -> tuple[list[dict], dict]:
    """build_group(g) -> 任意组上下文；answer_one(ctx, q) -> (hypothesis, meta)。

    meta.usage（若有）被汇总；meta 其余字段并入 record。返回 (records, usage_total)。
    """
    t0 = time.time()

    # 1) 建库（串行；V3 抽取内部已并行；单组失败重试 3 次后跳过并标记）
    ctxs = []
    for gi, g in enumerate(groups, 1):
        ctx = None
        for attempt in range(3):
            try:
                ctx = build_group(g)
                break
            except Exception as e:  # 网络抖动等单组故障不拖垮整轮
                import time as _t
                wait = 10.0 * (attempt + 1)
                with _print_lock:
                    print(f"[{tag}] 建库失败组 {gi}（{type(e).__name__}），{wait:.0f}s 后重试",
                          file=sys.stderr, flush=True)
                _t.sleep(wait)
        ctxs.append(ctx)
        if gi % 100 == 0:
            with _print_lock:
                print(f"[{tag}] 建库 {gi}/{len(groups)} 用时={time.time()-t0:.0f}s", flush=True)

    # 2) 展开并并行答题
    tasks = [(ctx, q) for ctx, g in zip(ctxs, groups) for q in g["questions"]]
    results: list = [None] * len(tasks)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    usage_lock = threading.Lock()
    done = 0

    def _work(i: int):
        ctx, q = tasks[i]
        hyp, meta = answer_one(ctx, q)
        return i, hyp, meta

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, i) for i in range(len(tasks))]
        for fut in as_completed(futs):
            i, hyp, meta = fut.result()
            u = meta.pop("usage", None)
            if u:
                with usage_lock:
                    usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
                    usage_total["completion_tokens"] += u.get("completion_tokens", 0)
            results[i] = (hyp, meta)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                with _print_lock:
                    print(f"[{tag}] 已答 {done}/{len(tasks)} 用时={time.time()-t0:.0f}s",
                          flush=True)

    # 3) 写出（顺序与输入一致）
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = out_dir / f"hyp_{tag}.jsonl"
    records = []
    with hyp_path.open("w", encoding="utf-8") as f:
        for (ctx, q), (hyp, meta) in zip(tasks, results):
            f.write(json.dumps({"question_id": q["question_id"], "hypothesis": hyp},
                               ensure_ascii=False) + "\n")
            records.append({"question_id": q["question_id"],
                            "question_type": q.get("question_type"),
                            "is_abs": q["question_id"].endswith("_abs"),
                            "answer": q.get("answer", ""), "hypothesis": hyp, **meta})
    (out_dir / f"{records_name}_{tag}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return records, usage_total
