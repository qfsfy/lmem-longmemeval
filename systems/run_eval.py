"""评测入口：统一驱动各版本记忆系统（v1/v2/v3）在 LongMemEval Oracle 上产出 JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from systems.v1_lexical import deepseek_config, load_env, local_config, run, stratified_subset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="LongMemEval Oracle 评测驱动")
    ap.add_argument("--system", default="v1", choices=["v1", "v2", "v3", "v4", "v5"])
    ap.add_argument("--backend", default="deepseek", choices=["deepseek", "local"],
                    help="local=llama.cpp Qwen3-4B（零成本统一口径）")
    ap.add_argument("--data", default="data/longmemeval/longmemeval_oracle.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--subset", type=int, default=0, help="分层抽样题数，0=全量500")
    ap.add_argument("--limit", type=int, default=0, help="取前 N 题（调试用）")
    ap.add_argument("--abstain-score", type=float, default=None)
    ap.add_argument("--cos-min", type=float, default=0.0, help="v2 稠密通道拒答阈值")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    load_env()
    questions = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.subset:
        questions = stratified_subset(questions, args.subset, args.seed)
    if args.limit:
        questions = questions[: args.limit]

    tag_parts = [args.system, "sub%d" % args.subset if args.subset else "full"]
    if args.backend == "local":
        tag_parts.append("loc")
    if args.seed != 42:
        tag_parts.append("seed%d" % args.seed)
    tag = "_".join(tag_parts)
    out = args.out or f"runs/{args.system}"

    model_cfg = deepseek_config() if args.backend == "deepseek" else local_config()
    if args.system == "v1":
        from systems.v1_lexical import run as run_v1
        kw = {} if args.abstain_score is None else {"abstain_score": args.abstain_score}
        run_v1(questions, out, tag=tag, model_cfg=model_cfg, **kw)
    elif args.system == "v2":
        from systems.v2_dense import run as run_v2
        gate = (0.0, 0.0) if args.abstain_score is None else (args.abstain_score, args.cos_min)
        run_v2(questions, out, tag=tag, model_cfg=model_cfg, abstain=gate)
    elif args.system == "v3":
        from systems.v3_layered import run as run_v3
        run_v3(questions, out, tag=tag, model_cfg=model_cfg)
    elif args.system == "v4":
        from systems.v4_route import run as run_v4
        run_v4(questions, out, tag=tag, model_cfg=model_cfg,
               use_reasoner=(args.backend == "deepseek"))
    elif args.system == "v5":
        from systems.v5_fusion import run as run_v5
        run_v5(questions, out, tag=tag, model_cfg=model_cfg,
               use_reasoner=(args.backend == "deepseek"))


if __name__ == "__main__":
    main()
