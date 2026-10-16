"""可插拔语义向量后端。

- ZhipuEmbedder：智谱 embedding-3（1024 维，批量 64，sqlite 逐文本缓存，跨重跑零成本）；
- LocalEmbedder：fastembed bge-small-en-v1.5（离线备用，零 API 成本）。

选择逻辑（make_embedder）：环境变量 EMBED_BACKEND=zhipu|fastembed|auto（默认 auto：
有 ZHIPU_EMBED_KEY 走智谱，否则本地）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

_CACHE_PATH = Path("runs/emb_cache.sqlite3")


class ZhipuEmbedder:
    """智谱 embedding-3：OpenAI 兼容 /embeddings，逐文本磁盘缓存。"""

    BATCH = 64

    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4",
                 model: str = "embedding-3", dimensions: int = 1024) -> None:
        from openai import OpenAI
        self.model = model
        self.dimensions = dimensions
        self.name = f"{model}-{dimensions}d"
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(_CACHE_PATH), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vec(h TEXT PRIMARY KEY, d INTEGER, v BLOB)")
        self._db.commit()

    def _hash(self, text: str) -> str:
        return hashlib.md5((self.name + "\x00" + text).encode("utf-8")).hexdigest()

    def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(8):
            try:
                r = self._client.embeddings.create(
                    model=self.model, input=texts, dimensions=self.dimensions)
                return [d.embedding for d in r.data]
            except Exception as e:
                name = type(e).__name__
                if attempt == 7:
                    raise
                wait = min(60.0, (2 ** attempt) * 2.0)
                print(f"[embed] {name}, backoff {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str]) -> "object":
        import numpy as np
        out: list = [None] * len(texts)
        missing: list[int] = []
        for i, t in enumerate(texts):
            row = self._db.execute("SELECT d, v FROM vec WHERE h=?", (self._hash(t),)).fetchone()
            if row is not None:
                out[i] = np.frombuffer(row[1], dtype=np.float32)
            else:
                missing.append(i)
        for start in range(0, len(missing), self.BATCH):
            idx = missing[start:start + self.BATCH]
            vecs = self._embed_remote([texts[i] for i in idx])
            for i, v in zip(idx, vecs):
                arr = np.asarray(v, dtype=np.float32)
                out[i] = arr
                self._db.execute("INSERT OR REPLACE INTO vec VALUES (?,?,?)",
                                 (self._hash(texts[i]), len(arr), arr.tobytes()))
        self._db.commit()
        return out

    def query_embed(self, questions: list[str]):
        return self.embed(questions)


class LocalEmbedder:
    """fastembed bge-small-en-v1.5（与 V2 原实现一致的离线路径）。"""

    def __init__(self) -> None:
        from fastembed import TextEmbedding
        self._m = TextEmbedding("BAAI/bge-small-en-v1.5")
        self.name = "bge-small-en-v1.5"

    def embed(self, texts: list[str]):
        import numpy as np
        vecs = []
        for i in range(0, len(texts), 64):
            vecs.extend(self._m.embed(texts[i:i + 64]))
        return [np.asarray(v, dtype=np.float32) for v in vecs]

    def query_embed(self, questions: list[str]):
        import numpy as np
        return [np.asarray(v, dtype=np.float32) for v in self._m.query_embed(questions)]


def make_embedder():
    import os
    backend = os.environ.get("EMBED_BACKEND", "auto")
    key = os.environ.get("ZHIPU_EMBED_KEY")
    if backend == "fastembed" or (backend == "auto" and not key):
        return LocalEmbedder()
    return ZhipuEmbedder(api_key=key)
