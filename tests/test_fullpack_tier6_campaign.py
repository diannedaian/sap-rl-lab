import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_fullpack_tier6_h40.py"
spec = importlib.util.spec_from_file_location("tier6_campaign_test", SCRIPT)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_setup_uses_full_tier6_and_40_in_both_models_and_scripted_pools():
    attributes = (
        "TIER",
        "CATALOG_ID",
        "SEED_OFFSET",
        "SEEDS",
        "VAL_SEED",
        "TEST_SEED",
        "configuration",
        "build_scripted_league",
    )
    old = {key: getattr(run.exp, key) for key in attributes}
    try:
        run.setup()
        catalog, game = run.exp.configuration().environment()
        assert catalog.catalog_id == "turtle-v0.46-full-v8"
        assert game.max_turns == 40 and game.max_shop_tier == 6
        assert game.max_actions_per_turn == 30
        with patch.object(run, "build_scripted_league") as builder:
            run.exp.build_scripted_league("full_stats", 2, 1, catalog=catalog)
        assert builder.call_args.kwargs["config"] == game
    finally:
        for key, value in old.items():
            setattr(run.exp, key, value)


@pytest.mark.parametrize("wins,forced,passed", [(400, 0, True), (399, 0, False), (450, 30, False)])
def test_screen_requires_three_models_3000_learned_games_and_reliability(
    tmp_path, wins, forced, passed
):
    folder = run.segment_dir(tmp_path, 1)
    folder.mkdir(parents=True)
    rows = [
        {
            "success": i < wins,
            "forced_end_turns": int(i < forced),
            "wins": 8,
            "truncated": False,
            "action_counts": {"buy_pet": 10},
            "return": 1.0,
        }
        for i in range(500)
    ]
    result = {
        "families": {
            name: {"episodes": 500, "episode_results": rows}
            for name in ("full_stats", "full_summon", "full_tempo", "test_a", "test_b")
        }
    }
    for seed in run.SEEDS:
        run.write_new(folder / f"test-{seed}.json", {"evaluation": result})
    summary = run.screen(tmp_path, 1)
    assert summary["learned_episodes"] == 3000
    assert summary["screen_passed"] == passed
    assert summary["confirmation_pending"] and not summary["goal_complete"]


def test_batch_dispatches_all_workers_to_fullpack40_script(tmp_path):
    child = Mock(pid=42)
    child.poll.return_value = 0
    with patch.object(run.subprocess, "Popen", return_value=child) as launch:
        run.batch(tmp_path, "candidate", run.SEEDS, 2)
    assert launch.call_count == 3
    assert launch.call_args.args[0][2] == str(SCRIPT)
    assert launch.call_args.args[0][-2:] == ["--segment", "2"]
    child.terminate.assert_not_called()


def test_seals_reject_modification(tmp_path):
    path = tmp_path / "test.json"
    run.seal(path, {"hello": 1})
    assert run.sealed(path)["hello"] == 1
    path.write_text(json.dumps({"hello": 2}))
    with pytest.raises(ValueError, match="changed"):
        run.sealed(path)


def test_campaign_keeps_training_after_a_normal_low_score_segment(tmp_path):
    completed = []
    reasons = {"stop_reason": "budget_complete"}
    stop = {"stop_reason": "sustained_severe_deterioration"}

    def reads(path):
        return reasons if "segment001" in str(path) else stop

    with (
        patch.object(run, "setup"),
        patch.object(run, "prepare"),
        patch.object(run, "batch"),
        patch.object(run.exp, "prepare_candidates"),
        patch.object(run, "read", side_effect=reads),
        patch.object(run, "selections", return_value=dict.fromkeys(map(str, run.SEEDS))),
        patch.object(run, "write_new"),
        patch.object(run, "prepare_segment", side_effect=lambda root, i: completed.append(i)),
    ):
        run.campaign(tmp_path, tmp_path / "curriculum")
    assert completed == [1, 2]
