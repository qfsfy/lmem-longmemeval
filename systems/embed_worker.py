"""嵌入工作子进程：stdin 收 JSON 文本数组，stdout 回 JSON 向量数组。

由 SubprocessEmbedder 每块调用一次；进程退出即释放全部 onnxruntime 原生内存，
主进程永不加载 ORT（根治变长 batch 下的原生内存碎片增长）。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EMBED_THREADS", "6")

import systems.embedders as em  # noqa: E402

texts = json.loads(sys.stdin.read())
kind = os.environ.get("WORKER_EMBEDDER", "onnx")
emb = em.FastembedEmbedder() if kind == "fastembed" else em.LocalEmbedder()
vecs = emb.embed(texts)
print(json.dumps({"name": emb.name, "vecs": [v.tolist() for v in vecs]}))
