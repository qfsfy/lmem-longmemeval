"""命令行入口：lmem ui / ask / chat / ingest / probe / eval。

设计原则：所有命令不加参数即可运行（Mock 模式），--config 可覆盖配置。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    help="LMem —— 长交互式AI记忆系统 harness（GLM-5.3 / GLM-5.3-Flash）",
    no_args_is_help=True, add_completion=False,
)
eval_app = typer.Typer(help="LongMemEval 评测", no_args_is_help=True, invoke_without_command=True)
app.add_typer(eval_app, name="eval")
console = Console()


def _harness(config: Optional[str]):
    from .pipeline.harness import MemoryHarness
    from .config import load_config

    h = MemoryHarness(load_config(config))
    mode = h.cfg.effective_backend()
    style = "yellow" if mode == "mock" else "green"
    console.print(f"[{style}]后端：{mode}（flash={h.cfg.flash.name}, "
                  f"flagship={h.cfg.flagship.name}）[/{style}]")
    return h


@app.command()
def ui(port: int = typer.Option(7860, help="端口"),
       host: str = typer.Option("127.0.0.1", help="监听地址"),
       config: Optional[str] = typer.Option(None, "--config", "-c")):
    """启动 Web 界面（对话 / 记忆库 / 设置）。"""
    from .ui.app import create_app

    demo = create_app(config)
    console.print(Panel.fit(
        f"[bold]LMem Web UI[/bold]\nhttp://{host}:{port}\n"
        "对话页可切换「提问 / 写入记忆」；「一键载入示例记忆」可快速体验。",
        border_style="blue"))
    demo.launch(server_name=host, server_port=port, inbrowser=True, show_error=True,
                theme=getattr(demo, "_lmem_theme", None))


@app.command()
def ask(question: str = typer.Argument(..., help="要问的问题"),
        as_of: Optional[str] = typer.Option(None, help="时间旅行：以该日期的视角回答"),
        config: Optional[str] = typer.Option(None, "--config", "-c")):
    """单问一次（先 ingest 或在 UI 里写入记忆）。"""
    h = _harness(config)
    ans = h.answer(question, as_of=as_of)
    console.print(Panel(Markdown(ans.answer), title="回答", border_style="cyan"))
    t = Table(show_header=False, box=None)
    for k, v in ans.meta_brief().items():
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def chat(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """交互式命令行对话（/write 前缀写入记忆，/stats /reset /exit）。"""
    h = _harness(config)
    console.print("[dim]命令：/write <内容> 写入记忆 ｜ /stats 统计 ｜ /reset 清空 ｜ /exit 退出[/dim]")
    while True:
        try:
            line = console.input("[bold cyan]你 ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/exit":
            break
        if line == "/stats":
            console.print_json(json.dumps(h.stats(), ensure_ascii=False))
            continue
        if line == "/reset":
            h.reset()
            console.print("[yellow]记忆已清空[/yellow]")
            continue
        if line.startswith("/write "):
            out = h.add_text(line[7:])
            console.print(f"[green]✅ +{out['cards_added']} 卡，失效 {out['superseded']}[/green]")
            continue
        ans = h.answer(line)
        console.print(Panel(Markdown(ans.answer), border_style="cyan"))
        console.print(f"[dim]{ans.qtype} ｜ route={ans.route} ｜ conf={ans.confidence:.2f} "
                      f"｜ coverage={ans.coverage:.2f} ｜ gate={ans.gate_action}[/dim]")


@app.command()
def ingest(path: Path = typer.Argument(..., help="会话 JSONL：{session_id,date,turns:[{role,content}]}"),
           config: Optional[str] = typer.Option(None, "--config", "-c")):
    """批量写入会话文件。"""
    from .memory.schema import Session, Turn

    h = _harness(config)
    total = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        s = Session(session_id=d["session_id"], date=d["date"],
                    turns=[Turn(role=t["role"], content=t["content"]) for t in d["turns"]])
        out = h.add_session(s)
        total += out["cards_added"]
        console.print(f"  [{i + 1}] {s.session_id}: +{out['cards_added']} 卡")
    console.print(f"[bold green]完成：共写入 {total} 张记忆卡[/bold green]")


@app.command()
def probe(context: int = typer.Option(0, help="附加长上下文探测的填充 token 数（0=跳过）"),
          config: Optional[str] = typer.Option(None, "--config", "-c")):
    """探测 GLM 端点连通性 / 时延 /（可选）长上下文能力。"""
    from .config import load_config
    from .probe import run_probe

    run_probe(load_config(config), context_tokens=context or None)


@eval_app.command("run")
def eval_run(
    data: Path = typer.Argument(..., help="LongMemEval 数据 json/jsonl 或目录"),
    out: Path = typer.Option("runs/eval", help="输出目录"),
    limit: Optional[int] = typer.Option(None, help="只跑前 N 题（调试用）"),
    judge: bool = typer.Option(False, "--judge", help="跑完用 judge 模型快速判分（非官方口径）"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """跑 LongMemEval 子集，输出官方格式 JSONL + 内部记录。"""
    from .evals.longmemeval import load_questions, run_eval
    from .pipeline.harness import MemoryHarness
    from .config import load_config

    questions = load_questions(data)
    console.print(f"加载 {len(questions)} 题")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = out / stamp
    harness = MemoryHarness(load_config(config))
    records = run_eval(harness, questions, out_dir / "hypotheses.jsonl",
                       limit=limit, verbose=True)
    (out_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(f"[green]已写出：[/green]{out_dir / 'hypotheses.jsonl'}")

    if judge:
        from .evals.judge import judge_one
        from .evals.metrics import format_report, summarize

        backend = harness.backend
        judged = []
        for r, q in zip(records, questions[: len(records)]):
            v = judge_one(backend, harness.cfg.judge, q["question"], r["answer"], r["hypothesis"])
            judged.append({"question_type": r["question_type"],
                           "verdict": v.verdict, "abstained": r["abstained"]})
        summary = summarize(records, judged)
        report = format_report(summary)
        console.print(Panel(Markdown(f"```\n{report}\n```"), title="快速判分（非官方口径）"))
        (out_dir / "report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
