"""通用并行评测流水线（v2：分块流式，内存有界）。

历史教训：一次把所有组的 store/DenseIndex 建在内存里，495 组 × (sqlite :memory:
+ BM25 词典 + 向量矩阵) 会累积数 GB，叠加 onnxruntime arena 碎片化直接把机器打爆
（2026-10-16 OOM 事故）。v2 改为按块处理：每 CHUNK 组建库 → 并行答题 → 立即落盘
→ 整块释放，内存峰值 ≈ CHUNK 内组数 × 单组开销。
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
CHUNK_GROUPS = 40


def run_pipeline(
    groups: list[dict],
    build_group: Callable,
    answer_one: Callable,
    out_dir: str | Path,
    tag: str,
    records_name: str = "records",
    workers: int = 2,
    chunk_groups: int = CHUNK_GROUPS,
) -> tuple[list[dict], dict]:
    """build_group(g) -> 组上下文；answer_one(ctx, q) -> (hypothesis, meta)。

    meta.usage（若有）被汇总；meta 其余字段并入 record。返回 (records, usage_total)。
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = out_dir / f"hyp_{tag}.jsonl"
    records: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    usage_lock = threading.Lock()
    done_total = 0

    with hyp_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(groups), chunk_groups):
            chunk = groups[start:start + chunk_groups]

            # 建库（串行；抽取/嵌入的内部并发由各组自身控制）
            ctxs = []
            for g in chunk:
                ctx = None
                for attempt in range(3):
                    try:
                        ctx = build_group(g)
                        break
                    except Exception as e:  # 网络抖动等单组故障不拖垮整轮
                        wait = 10.0 * (attempt + 1)
                        with _print_lock:
                            print(f"[{tag}] 建库失败（{type(e).__name__}），{wait:.0f}s 后重试",
                                  file=sys.stderr, flush=True)
                        time.sleep(wait)
                ctxs.append(ctx)

            # 组内答题并行（打到推理服务的多个 slot）
            def _answer_chunk():
                chunk_results = []
                for ctx, g in zip(ctxs, chunk):
                    if ctx is None:  # 建库失败：按拒答输出，不中断
                        for q in g["questions"]:
                            chunk_results.append((q, "I don't have that information in my memory.",
                                                  {"qtype": "error", "error": "build failed"}))
                        continue

                    def one(q, ctx=ctx):
                        try:
                            return q, *answer_one(ctx, q)
                        except Exception as e:
                            return q, "I don't have that information in my memory.", \
                                {"qtype": "error", "error": str(e)[:200]}

                    if workers > 1 and len(g["questions"]) > 1:
                        with ThreadPoolExecutor(max_workers=workers) as ex:
                            chunk_results.extend(ex.map(one, g["questions"]))
                    else:
                        chunk_results.extend(one(q) for q in g["questions"])
                return chunk_results

            chunk_results = _answer_chunk()

            for q, hyp, meta in chunk_results:
                f.write(json.dumps({"question_id": q["question_id"], "hypothesis": hyp},
                                   ensure_ascii=False) + "\n")
                u = meta.pop("usage", None)
                if u:
                    with usage_lock:
                        usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
                        usage_total["completion_tokens"] += u.get("completion_tokens", 0)
                records.append({"question_id": q["question_id"],
                                "question_type": q.get("question_type"),
                                "is_abs": q["question_id"].endswith("_abs"),
                                "answer": q.get("answer", ""), "hypothesis": hyp, **meta})
            done_total += len(chunk_results)
            f.flush()

            # 整块释放（显式断开 sqlite :memory: 连接）
            for ctx in ctxs:
                try:
                    if hasattr(ctx, "close"):
                        ctx.close()
                    elif isinstance(ctx, tuple):
                        for part in ctx:
                            if hasattr(part, "close"):
                                part.close()
                except Exception:
                    pass
            del ctxs, chunk_results

            print(f"[{tag}] 已答 {done_total}/{sum(len(g['questions']) for g in groups)} "
                  f"用时={time.time()-t0:.0f}s", flush=True)

    (out_dir / f"{records_name}_{tag}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return records, usage_total
