"""通用文本工具：分词与日期规范化（无第三方依赖）。"""

from __future__ import annotations

import re
from datetime import date

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z0-9]+")
_DATE_RE = re.compile(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})")


def tokenize(text: str) -> list[str]:
    """中英混合分词：拉丁词元 + 中文二字组 + 中文单字。

    单字粒度保证跨表述的召回（问题"住在" vs 陈述"居住地"），
    区分度主要由 BM25 的 IDF 承担。
    """
    tokens = [w.lower() for w in _LATIN_RE.findall(text)]
    cjk = _CJK_RE.findall(text)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    tokens.extend(cjk)
    return tokens


def norm_date(text: str) -> str | None:
    """从任意文本中提取 YYYY-MM-DD；失败返回 None。"""
    m = _DATE_RE.search(text or "")
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def today_str() -> str:
    return date.today().isoformat()
