"""V3 遗忘与冲突机制测试（离线，不调 LLM：直接构造原子）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmem.memory.schema import Session, Turn
from lmem.memory.store import MemoryStore
from systems.v3_layered import AtomStore, atom_text


def make_atom_store(tmp_path, capacity=5) -> AtomStore:
    store = MemoryStore(":memory:")
    atoms = AtomStore(store)
    atoms.CAPACITY = capacity
    return atoms


def _session(sid, date, content) -> Session:
    return Session(sid, date, [Turn("user", content)])


def test_supersession_keeps_history(tmp_path):
    atoms = make_atom_store(tmp_path)
    atoms.add_atom({"predicate": "lives_in", "object": "Austin", "kind": "profile",
                    "date": "2024-01-05", "quote": "I live in Austin", "importance": 0.9},
                   _session("s1", "2024-01-05", "I live in Austin"))
    atoms.add_atom({"predicate": "lives_in", "object": "Denver", "kind": "profile",
                    "date": "2024-05-01", "quote": "I moved to Denver", "importance": 0.9},
                   _session("s2", "2024-05-01", "I moved to Denver"))
    all_cards = atoms.store.all_cards()
    active = [c for c in all_cards if c.valid_to is None]
    old = [c for c in all_cards if c.valid_to is not None]
    assert len(active) == 1 and active[0].value == "Denver"
    assert len(old) == 1 and old[0].value == "Austin" and old[0].superseded_by
    # as-of 查询：2024-03 时旧住址仍有效
    past = [c for c in atoms.store.active_at("2024-03-01") if c.predicate == "lives_in"]
    assert len(past) == 1 and past[0].value == "Austin"


def test_dedupe_bumps_importance(tmp_path):
    atoms = make_atom_store(tmp_path)
    s = _session("s1", "2024-01-05", "I love hiking")
    a = {"predicate": "likes", "object": "hiking", "kind": "preference",
         "date": "", "quote": "I love hiking", "importance": 0.5}
    first = atoms.add_atom(a, s)
    again = atoms.add_atom(dict(a), s)
    assert first and again is None  # 第二次被去重
    row = atoms.store.conn.execute("SELECT importance FROM atom_meta WHERE id=?",
                                   (first,)).fetchone()
    assert abs(row[0] - 0.55) < 1e-6


def test_capacity_forget_evicts_lowest_score(tmp_path):
    atoms = make_atom_store(tmp_path, capacity=3)
    # 4 个原子：1 个重要（0.9）、1 个高访问、2 个低价值
    items = [
        ("bought", "house", 0.9, 0),
        ("bought", "car", 0.5, 9),      # 高访问
        ("mentioned", "spam brand", 0.1, 0),
        ("mentioned", "random fact", 0.1, 0),
    ]
    for i, (p, o, imp, _) in enumerate(items):
        atoms.add_atom({"predicate": p, "object": o, "kind": "event", "date": "2024-01-0%d" % (i + 1),
                        "quote": f"{p} {o}", "importance": imp},
                       _session("s%d" % i, "2024-01-0%d" % (i + 1), f"{p} {o}"))
    # 模拟 "car" 被高频访问
    atoms.store.conn.execute(
        "UPDATE atom_meta SET access_count=9 WHERE id="
        "(SELECT id FROM cards WHERE value='car')")
    atoms.store.conn.commit()
    # 再写入第 5 个触发容量淘汰（CAPACITY=3）
    atoms.add_atom({"predicate": "bought", "object": "lamp", "kind": "event",
                    "date": "2024-06-01", "quote": "bought lamp", "importance": 0.6},
                   _session("s9", "2024-06-01", "bought a lamp"))
    remaining = {c.value for c in atoms.store.all_cards()}
    assert len(remaining) == 3
    assert "house" in remaining      # 高重要度保留
    assert "car" in remaining        # 高访问保留
    assert "lamp" in remaining       # 新写入保留
    assert "spam brand" not in remaining or "random fact" not in remaining
