"""可插拔语义向量后端。

- ZhipuEmbedder：智谱 embedding-3（1024 维，批量 64，sqlite 逐文本缓存，跨重跑零成本）；
- LocalEmbedder：fastembed bge-small-en-v1.5（离线备用，零 API 成本）。

选择逻辑（make_embedder）：环境变量 EMBED_BACKEND=zhipu|fastembed|auto（默认 auto：
有 ZHIPU_EMBED_KEY 走智谱，否则本地）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
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
    """bge-small-en-v1.5 直连 ONNX（不经 fastembed）。

    2026-10-16 事故复盘：onnxruntime 的 BFC arena 在变长 batch 下单调增长不释放
    （多次 embed 后进程 RSS 达 8GB+）。这里直接建 Session 并设
    enable_cpu_mem_arena=False，显存即时归还、内存有界；mean pooling + L2
    与 fastembed 对该模型的后处理保持一致。
    """

    BATCH = 32
    MAX_LEN = 192  # 记忆卡均值 ~30 token、上限 60 词；192 足够，且把注意力激活峰值压低 ~7 倍

    def __init__(self) -> None:
        import os
        import onnxruntime as ort
        from tokenizers import Tokenizer

        local = Path(os.environ.get(
            "FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")))
        snaps = list(local.glob("models--Qdrant--bge-small-en-v1.5-onnx-Q/snapshots/*"))
        if not snaps:
            raise FileNotFoundError("bge-small onnx 模型未找到，请先运行一次 fastembed 下载")
        snap = snaps[0]
        self.name = "bge-small-en-v1.5-onnx"

        self._tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
        self._tok.enable_truncation(max_length=self.MAX_LEN)

        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False  # 关键：arena 关闭，防止内存无限增长
        so.intra_op_num_threads = int(os.environ.get("EMBED_THREADS", "4"))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(snap / "model_optimized.onnx"),
                                         sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self.sess.get_inputs()}

    def _forward(self, texts: list[str]):
        import numpy as np
        enc = self._tok.encode_batch(texts)
        maxlen = max(len(e.ids) for e in enc)
        ids = np.zeros((len(enc), maxlen), dtype=np.int64)
        mask = np.zeros((len(enc), maxlen), dtype=np.int64)
        for i, e in enumerate(enc):
            n = len(e.ids)
            ids[i, :n] = e.ids
            mask[i, :n] = e.attention_mask
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feeds)[0]  # (B, L, H) last_hidden_state
        mask_f = mask[:, :, None].astype(np.float32)
        summed = (out * mask_f).sum(axis=1)
        counts = np.clip(mask_f.sum(axis=1), 1e-9, None)
        mean = summed / counts
        norm = np.linalg.norm(mean, axis=1, keepdims=True)
        return mean / np.maximum(norm, 1e-9)

    def embed(self, texts: list[str]):
        vecs = []
        for i in range(0, len(texts), self.BATCH):
            vecs.extend(self._forward(texts[i:i + self.BATCH]))
        return vecs

    def query_embed(self, questions: list[str]):
        return self.embed(questions)


def make_embedder():
    import os
    import sys
    backend = os.environ.get("EMBED_BACKEND", "auto")
    key = os.environ.get("ZHIPU_EMBED_KEY")
    if backend == "fastembed":
        # 本地 ONNX 走子进程隔离：onnxruntime 的原生分配在 Windows 上随变长 batch
        # 碎片化增长（进程 RSS 可达 8GB+ 且 arena=False 也无法阻止），子进程每批
        # 退出一次即彻底释放，主进程只保留向量本身。
        return SubprocessEmbedder()
    if backend == "zhipu" or (backend == "auto" and key):
        return ZhipuEmbedder(api_key=key)
    return SubprocessEmbedder()


class SubprocessEmbedder:
    """LocalEmbedder 的子进程隔离封装（每批一个子进程，退出即释放原生内存）。"""

    BATCH = 1024

    def __init__(self) -> None:
        import subprocess
        self._python = sys.executable
        self._worker = str(Path(__file__).resolve().parent / "embed_worker.py")
        self.name = "bge-small-en-v1.5-onnx"

    def embed(self, texts: list[str]):
        out: list = []
        env = dict(os.environ)
        env["FASTEMBED_CACHE_PATH"] = env.get(
            "FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
        for start in range(0, len(texts), self.BATCH):
            batch = texts[start:start + self.BATCH]
            r = subprocess.run(
                [self._python, self._worker],
                input=json.dumps(batch), capture_output=True, text=True,
                encoding="utf-8", timeout=600, env=env)
            if r.returncode != 0:
                raise RuntimeError(f"embed worker failed: {r.stderr[-400:]}")
            payload = json.loads(r.stdout.strip().splitlines()[-1])
            out.extend(payload["vecs"])
        return out

    def query_embed(self, questions: list[str]):
        return self.embed(questions)
