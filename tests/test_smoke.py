"""冒烟测试：全流程离线运行（Mock 后端），不依赖网络与 API Key。"""

import json
import os

import pytest

# 强制离线，防止本机环境变量影响
for k in ("ZAI_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(k, None)

from lmem.config import HarnessConfig
from lmem.evals.longmemeval import run_eval
from lmem.memory.schema import Session, Turn
from lmem.pipeline.harness import MemoryHarness


def make_harness(tmp_path) -> MemoryHarness:
    cfg = HarnessConfig(backend="mock", store_path=str(tmp_path / "smoke.sqlite3"))
    return MemoryHarness(cfg)


def demo_harness(tmp_path) -> MemoryHarness:
    h = make_harness(tmp_path)
    h.add_session(Session("s1", "2024-01-10", [
        Turn("user", "我叫小明，我住在北京，我喜欢蓝色。"),
        Turn("assistant", "好的，记住了。"),
    ]))
    h.add_session(Session("s2", "2024-03-01", [
        Turn("user", "我搬到上海了，我现在喜欢红色。"),
    ]))
    return h


def test_ingest_extracts_cards(tmp_path):
    h = make_harness(tmp_path)
    out = h.add_session(Session("s1", "2024-01-10", [
        Turn("user", "我叫小明，我喜欢蓝色。"),
    ]))
    assert out["cards_added"] >= 2
    st = h.stats()
    assert st["cards"] >= 2
    assert st["sessions"] == 1


def test_conflict_supersedes_old_card(tmp_path):
    h = demo_harness(tmp_path)
    lives = [c for c in h.store.all_cards() if c.predicate == "lives_in"]
    assert len(lives) == 2
    old = next(c for c in lives if c.value and "北京" in c.value)
    new = next(c for c in lives if c.value and "上海" in c.value)
    assert old.valid_to == new.observed_at and old.superseded_by == new.id
    # as-of 时间旅行：3 月看上海有效，1 月看北京有效
    assert not old.active_at("2024-03-01") and old.active_at("2024-01-15")
    assert new.active_at("2024-03-01")


def test_answer_current_state(tmp_path):
    h = demo_harness(tmp_path)
    ans = h.answer("我现在住在哪里？", as_of="2024-03-05")
    assert not ans.abstained
    assert "上海" in ans.answer
    assert ans.citations, "应有引用编号"


def test_answer_historical_state(tmp_path):
    h = demo_harness(tmp_path)
    old = h.answer("我现在住在哪里？", as_of="2024-01-15")
    assert "北京" in old.answer
    past = h.answer("我以前喜欢什么颜色？", as_of="2024-03-05")
    assert "蓝" in past.answer


def test_abstention_gate(tmp_path):
    h = make_harness(tmp_path)
    h.add_text("我叫小明。")
    ans = h.answer("我的银行卡密码是多少？")
    assert ans.abstained and ans.gate_action == "abstain_coverage"


def test_cascade_route_flag(tmp_path):
    h = demo_harness(tmp_path)
    h.cfg.router.strategy = "by_type"  # 静态路由，行为可预测
    ans = h.answer("我现在住在哪里？", as_of="2024-03-05")
    assert ans.route in ("flash", "flagship")


def test_eval_roundtrip(tmp_path):
    h = demo_harness(tmp_path)
    questions = [{
        "question_id": "q1",
        "question_type": "knowledge_update",
        "question": "我现在住在哪里？",
        "answer": "上海",
        "question_date": "2024-03-05",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2024-01-10", "2024-03-01"],
        "haystack_sessions": [
            [{"role": "user", "content": "我住在北京。"}],
            [{"role": "user", "content": "我搬到上海了。"}],
        ],
    }]
    out = tmp_path / "hypo.jsonl"
    records = run_eval(h, questions, out, verbose=False)
    assert len(records) == 1
    line = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert line["question_id"] == "q1" and "hypothesis" in line
    assert not records[0]["abstained"]


def test_reset(tmp_path):
    h = demo_harness(tmp_path)
    h.reset()
    assert h.stats()["cards"] == 0 and h.stats()["sessions"] == 0
