import importlib
from pathlib import Path


def test_exploration_replays_prefer_empty_loss_and_deduplicate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    audit = importlib.import_module("audit_exploration_pilot")
    rows = [
        {
            "seed": 1,
            "success": False,
            "wins": 2,
            "action_counts": {"buy_pet": 3},
            "forced_end_turns": 1,
            "truncated": False,
        },
        {
            "seed": 2,
            "success": False,
            "wins": 0,
            "action_counts": {"end_turn": 6},
            "forced_end_turns": 0,
            "truncated": False,
        },
        {
            "seed": 3,
            "success": True,
            "wins": 10,
            "action_counts": {"buy_pet": 5},
            "forced_end_turns": 0,
            "truncated": False,
        },
    ]
    cases = audit.choose_cases({"a": {"episode_results": rows}})
    assert {c["expected"]["seed"] for c in cases} == {1, 2, 3}
    assert len(audit.choose_cases({"a": {"episode_results": [rows[0]]}})) == 1
    assert audit.choose_cases({}) == []
