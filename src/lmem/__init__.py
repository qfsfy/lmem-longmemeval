"""lmem —— 长交互式AI记忆系统 harness。

对外主入口：MemoryHarness（add_session / answer）。
"""

__version__ = "0.1.0"

from .pipeline.harness import MemoryHarness, HarnessAnswer  # noqa: F401
from .config import load_config  # noqa: F401
