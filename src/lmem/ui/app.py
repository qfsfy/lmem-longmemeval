"""Gradio Web UI —— 易用性设计要点：

- 开箱即用：无 Key 时自动进入"演示模式"，一键载入示例记忆即可体验完整流程；
- 透明可解释：每次回答展示路由（flash/5.3）、置信度、覆盖率、门控动作与引用证据；
- 记忆可视化：记忆库标签页可搜索/筛选/查看失效状态，理解 harness"记得什么、忘了什么"。
"""

from __future__ import annotations

from typing import Optional

from ..config import load_config
from ..pipeline.harness import MemoryHarness

DEMO_SESSIONS = [
    ("2024-01-10", "s-demo-1",
     "我叫小明，今年29岁。我住在北京，是一名软件工程师。我特别喜欢蓝色的毛衣，周末经常去爬山。"),
    ("2024-03-02", "s-demo-2",
     "跟你说个事，我上个月搬到上海了，新工作是数据科学家。现在我更喜欢穿红色的外套，爬山改成了跑步。"),
]

DEMO_QUESTIONS = [
    "我叫什么名字？",
    "我现在住在哪里？",
    "我现在最喜欢穿什么？",
    "我以前喜欢穿什么颜色？",
    "我做过哪些工作？",
]


def create_app(config_path: Optional[str] = None):
    import gradio as gr

    harness = MemoryHarness(load_config(config_path))
    mock_mode = harness.cfg.effective_backend() == "mock"

    badge = "🧪 演示模式（未检测到 API Key，规则 Mock 后端）" if mock_mode else \
        f"🟢 已连接：{harness.cfg.flash.name} + {harness.cfg.flagship.name}"

    # ---------------------------------------------------------------- 回调

    def respond(message: str, mode: str, history: list):
        history = history or []
        if not message or not message.strip():
            return history, gr.update(), ""
        if mode == "✍️ 写入记忆":
            out = harness.add_text(message.strip())
            note = f"✅ 已写入 {out['cards_added']} 张记忆卡"
            if out["superseded"]:
                note += f"，并失效 {out['superseded']} 张冲突旧卡（知识更新）"
            history += [{"role": "user", "content": message},
                        {"role": "assistant", "content": note}]
            return history, gr.update(value=None), ""

        ans = harness.answer(message.strip())
        meta = ans.meta_brief()
        if mock_mode:
            meta["note"] = "Mock 演示输出"
        history += [{"role": "user", "content": message},
                    {"role": "assistant", "content": ans.answer}]
        rows = ["| # | 记忆内容 | 有效期 | 来源会话 |", "|---|---|---|---|"]
        for e in ans.evidence[:8]:
            valid = e.get("valid_to") or "仍有效"
            rows.append(f"| {e.get('cite')} | {e.get('content')} | "
                        f"{e.get('valid_from')}~{valid} | {e.get('session_id')} |")
        ev_md = "\n".join(rows) if ans.evidence else "_（无证据 → 走拒答门控）_"
        return history, meta, ev_md

    def load_demo() -> str:
        notes = []
        for date, sid, text in DEMO_SESSIONS:
            out = harness.add_text(text, session_id=sid, date=date)
            notes.append(f"{date}: +{out['cards_added']}卡")
            if out["superseded"]:
                notes[-1] += f"（失效{out['superseded']}）"
        return "已载入示例记忆（" + "；".join(notes) + "）。试试右边的示例问题吧！"

    def refresh_memory(query: str, kind: str, include_invalid: bool):
        cards = harness.store.all_cards(include_invalid=True)
        if not include_invalid:
            cards = [c for c in cards if c.valid_to is None]
        if kind != "全部":
            cards = [c for c in cards if c.kind == kind]
        if query and query.strip():
            q = query.strip()
            cards = [c for c in cards if q in c.content or q in c.value or q in c.predicate]
        rows = [[c.id, c.kind, c.content, c.valid_from, c.valid_to or "仍有效",
                 c.session_id, round(c.confidence, 2)] for c in cards]
        st = harness.store.stats()
        stats = (f"会话 **{st['sessions']}** ｜ 记忆卡 **{st['cards']}** "
                 f"（生效 {st['active']} / 已失效 {st['cards'] - st['active']}）")
        return rows, stats

    def config_md() -> str:
        s = harness.stats()
        return f"""### 当前配置
| 项 | 值 |
|---|---|
| 后端 | {s['backend']}{'（演示模式：无需 API Key）' if mock_mode else ''} |
| Flash 模型 | `{s['models']['flash']}` |
| Flagship 模型 | `{s['models']['flagship']}` |
| 路由策略 | {s['router']} |
| 记忆库 | {s['cards']} 卡 / {s['sessions']} 会话 |

### 切换到真实 GLM 模型
1. `pip install -e ".[ui]"` 已装好后，设置环境变量 `ZAI_API_KEY`（见 `.env.example`）；
2. 端点默认智谱 `open.bigmodel.cn`，Z.ai 用户改 `configs/default.yaml` 中 `base_url`；
3. 重启 `lmem ui`，徽标变为 🟢 即接入成功。

### 命令行同等能力
```bash
lmem ask "我现在住在哪里？"        # 单问
lmem chat                          # 交互式命令行
lmem ingest data/sessions.jsonl    # 批量写入
lmem eval run data/longmemeval/longmemeval_s.json --limit 20
lmem probe                         # 实测模型端点
```"""

    # ---------------------------------------------------------------- 布局

    import gradio.themes as themes

    with gr.Blocks(title="LMem · 长交互式AI记忆系统") as demo:
        gr.Markdown(f"# 🧠 LMem 记忆 Harness\n{badge}")

        with gr.Tab("💬 对话"):
            chatbot = gr.Chatbot(height=400,
                                 label="对话（写入模式下你说的每句话都会被记住）")
            with gr.Row():
                mode = gr.Radio(choices=["💬 提问", "✍️ 写入记忆"], value="💬 提问",
                                show_label=False, scale=1)
                msg = gr.Textbox(placeholder="输入问题（提问），或输入自我介绍（写入记忆）…",
                                 show_label=False, autofocus=True, scale=6)
                send = gr.Button("发送", variant="primary", scale=1)
            with gr.Accordion("🔍 回答详情（路由 / 置信度 / 门控 / 引用证据）", open=False):
                meta_json = gr.JSON(label="元信息")
                evidence_md = gr.Markdown()
            with gr.Row():
                demo_btn = gr.Button("🎬 一键载入示例记忆", variant="secondary")
                clear_btn = gr.Button("🗑️ 清空对话")
            gr.Markdown("**示例问题（先载入示例记忆）：** " +
                        " ｜ ".join(f"`{q}`" for q in DEMO_QUESTIONS))
            status = gr.Markdown("")

            inputs = [msg, mode, chatbot]
            outputs = [chatbot, meta_json, evidence_md]
            send.click(respond, inputs, outputs)
            msg.submit(respond, inputs, outputs)
            msg.submit(lambda: "", None, [msg])
            send.click(lambda: "", None, [msg])
            clear_btn.click(lambda: [], None, [chatbot])
            demo_btn.click(load_demo, None, [status])

        with gr.Tab("📚 记忆库"):
            stats_md = gr.Markdown()
            with gr.Row():
                search = gr.Textbox(label="搜索内容", placeholder="关键词…")
                kind = gr.Dropdown(choices=["全部", "profile", "fact", "summary"],
                                   value="全部", label="类型")
                include_invalid = gr.Checkbox(value=True, label="包含已失效记忆")
            table = gr.Dataframe(
                headers=["ID", "类型", "内容", "生效起", "生效止", "来源会话", "置信度"],
                interactive=False, wrap=True)
            refresh_btn = gr.Button("🔄 刷新")
            for comp in (refresh_btn, search, kind, include_invalid):
                comp.change(refresh_memory, [search, kind, include_invalid], [table, stats_md])

        with gr.Tab("⚙️ 设置与说明"):
            gr.Markdown(config_md)

        demo.load(refresh_memory, [search, kind, include_invalid], [table, stats_md])

    # Gradio 6：theme 在 launch() 时传入
    demo._lmem_theme = themes.Soft(primary_hue="indigo")
    return demo
