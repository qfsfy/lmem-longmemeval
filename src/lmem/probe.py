"""模型探针：实测 GLM 端点的可用性、时延与思考模式行为。

用法：lmem probe            # 基础连通性 + 时延
      lmem probe --context 200000   # 追加长上下文探测（耗 token，谨慎使用）
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

from .config import HarnessConfig


def run_probe(cfg: HarnessConfig, context_tokens: int | None = None) -> None:
    console = Console()
    backend = cfg.effective_backend()
    console.print(f"[bold]后端模式：[/bold]{backend}")
    if backend == "mock":
        console.print("[yellow]未检测到 API Key，当前为 Mock 演示模式，跳过网络探测。[/yellow]")
        console.print("配置方法见 configs/default.yaml 与 .env.example。")
        return

    from .llm.client import OpenAICompatBackend

    be = OpenAICompatBackend()
    table = Table(title="模型连通性探测")
    table.add_column("角色")
    table.add_column("模型")
    table.add_column("端点")
    table.add_column("首包时延", justify="right")
    table.add_column("输出抽样")
    for role, mcfg in (("flash", cfg.flash), ("flagship", cfg.flagship), ("judge", cfg.judge)):
        try:
            t0 = time.time()
            comp = be.complete(
                task="ping", model_cfg=mcfg,
                system="你是一个回声器。", user="请只回复两个字：就绪",
                max_tokens=16, temperature=0.0,
            )
            table.add_row(role, mcfg.name, mcfg.base_url,
                          f"{comp.latency_ms} ms", comp.text[:20])
        except Exception as e:  # noqa: BLE001
            table.add_row(role, mcfg.name, mcfg.base_url, "-", f"[red]{type(e).__name__}: {e}[/red]")
    console.print(table)

    if context_tokens:
        console.print(f"[bold]长上下文探测（≈{context_tokens} tokens 填充）：[/bold]")
        filler = "记忆 " * context_tokens  # 中文字符≈1 token/字（粗略）
        marker = f"暗号{int(time.time()) % 10000}"
        probe_cfg = cfg.flagship
        try:
            comp = be.complete(
                task="ping", model_cfg=probe_cfg,
                system="在大量填充文本中找出暗号并原样输出。",
                user=f"{filler}\n暗号是 {marker}，请只输出暗号本身。",
                max_tokens=32, temperature=0.0,
            )
            ok = marker in comp.text
            console.print(f"  模型 {probe_cfg.name}：{'✅ 通过' if ok else '❌ 失败'}"
                          f"（{comp.latency_ms} ms，输出：{comp.text[:50]}）")
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]失败：{type(e).__name__}: {e}[/red]")
        console.print("[dim]注意：填充 token 数为估算；精确窗口上限请二分探测。[/dim]")
