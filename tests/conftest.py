"""pytest 路径引导：src/ 与仓库根目录加入 sys.path（不依赖 pip -e 安装）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
