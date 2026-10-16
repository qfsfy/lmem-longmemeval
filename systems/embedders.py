"""可插拔语义向量后端（V6.1 修订）。

三个实现（EMBED_BACKEND 选择）：
- fastembed（默认，= V5 精确路径）：官方 fastembed bge-small-en-v1.5，与 V2/V3/V5
  全量结果完全同分布；其原生内存在分块 runner 下实测可完成全量（V3/V5 先例）。
- onnx：直连 ONNX + arena 关闭（MAX_LEN 192 / BATCH 32）——内存有界但截断与
  分词细节与 fastembed 存在分布差异，V6.0 全量 0.838 证明会损失检索质量，仅备用。
- onnx_sub：同上但子进程隔离（彻底防碎片增长），同样仅备用。
- zhipu：智谱 embedding-3 API（逐文本 sqlite 缓存）；当前 key 持续 RPM 过低
  （~17 向量/分钟）不足以全量，换高配额 key 即可启用。
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


class FastembedEmbedder:
    """V5 精确路径：fastembed bge-small-en-v1.5（batch 256、512 截断，均与 V5 相同）。"""

    BATCH = 32  # batch 256 x seq512 的注意力瞬时分配 3.2GB/层, 会打出 9GB RSS 尖峰;
    # mask 感知 mean pooling 下批大小不影响结果, 降批安全

    def __init__(self) -> None:
        from fastembed import TextEmbedding
        self._m = TextEmbedding("BAAI/bge-small-en-v1.5")
        self.name = "bge-small-en-v1.5"

    def embed(self, texts: list[str]):
        import numpy as np
        vecs = []
        for i in range(0, len(texts), self.BATCH):
            vecs.extend(self._m.embed(texts[i:i + self.BATCH]))
        return [np.asarray(v, dtype=np.float32) for v in vecs]

    def query_embed(self, questions: list[str]):
        import numpy as np
        return [np.asarray(v, dtype=np.float32) for v in self._m.query_embed(questions)]


class DirectOnnxEmbedder:
    """直连 ONNX + arena 关闭（备用：内存有界，但与 fastembed 分布存在差异）。"""

    BATCH = 32
    MAX_LEN = 192

    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        local = Path(os.environ.get(
            "FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")))
        snaps = list(local.glob("models--Qdrant--bge-small-en-v1.5-onnx-Q/snapshots/*"))
        if not snaps:
            raise FileNotFoundError("bge-small onnx 模型未找到")
        snap = snaps[0]
        self.name = "bge-small-en-v1.5-onnx"

        self._tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
        self._tok.enable_truncation(max_length=self.MAX_LEN)

        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
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
        out = self.sess.run(None, feeds)[0]
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


class SubprocessEmbedder:
    """DirectOnnxEmbedder 的子进程隔离封装（备用：根治原生内存碎片增长）。"""

    BATCH = 1024

    def __init__(self, worker_embedder: str = "onnx") -> None:
        self._python = sys.executable
        self._worker = str(Path(__file__).resolve().parent / "embed_worker.py")
        self._worker_embedder = worker_embedder
        self.name = "bge-small-en-v1.5" if worker_embedder == "fastembed"             else "bge-small-en-v1.5-onnx"

    def embed(self, texts: list[str]):
        out: list = []
        env = dict(os.environ)
        env["WORKER_EMBEDDER"] = self._worker_embedder
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


class ZhipuEmbedder:
    """智谱 embedding-3（1024 维，逐文本 sqlite 缓存）。"""

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

    def embed(self, texts: list[str]):
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


def make_embedder():
    """默认 = FastembedEmbedder（V5 精确路径，保证与已发布结果同分布）。"""
    import os
    backend = os.environ.get("EMBED_BACKEND", "fastembed")
    if backend == "zhipu":
        key = os.environ.get("ZHIPU_EMBED_KEY")
        if not key:
            raise RuntimeError("EMBED_BACKEND=zhipu 需要 ZHIPU_EMBED_KEY")
        return ZhipuEmbedder(api_key=key)
    if backend == "onnx":
        return DirectOnnxEmbedder()
    if backend == "onnx_sub":
        return SubprocessEmbedder(worker_embedder="onnx")
    # fastembed（默认）也走子进程隔离：其 ORT session 的 arena 在变长 batch 下
    # 同样会碎片化增长（实测单进程 RSS 8GB+），子进程退出即彻底释放
    return SubprocessEmbedder(worker_embedder="fastembed")
