"""统一入口（绕开中文路径下 pip install -e 的构建问题）：

    python run.py eval  --system v1 --subset 50      # 记忆系统评测 → JSONL + records
    python run.py judge runs/v1/hyp_v1_sub50.jsonl   # 官方口径判分
    python run.py test                               # 离线冒烟测试
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if cmd == "eval":
        from systems.run_eval import main as eval_main
        eval_main(rest)
    elif cmd == "judge":
        from systems.judge_official import main as judge_main
        sys.argv = ["judge"] + rest
        judge_main()
    elif cmd == "test":
        import pytest
        raise SystemExit(pytest.main(["tests", "-q"]))
    else:
        print(__doc__)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
