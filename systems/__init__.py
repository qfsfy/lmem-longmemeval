"""systems 包：各版本记忆系统实现。

导入本包时自动把 src/ 与仓库根目录加入 sys.path，
使 `python -m systems.<mod>` 无需安装即可运行（规避中文路径下 pip -e 的构建问题）。
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
