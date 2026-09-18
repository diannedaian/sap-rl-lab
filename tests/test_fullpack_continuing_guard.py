import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

from sap_rl_lab.fullpack.ppo_guardrails import GuardrailConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/continue_fullpack40.py"
spec = importlib.util.spec_from_file_location("continuation_test", SCRIPT)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def suite(wins=15, force=0, truncated=False):
    rows = [
        {
            "success": i < wins,
            "forced_end_turns": int(i < force),
            "wins": 8,
            "truncated": truncated and i == 0,
            "action_counts": {"buy_pet": 10},
            "return": 1.0,
        }
        for i in range(20)
    ]
    return {
        "families": {
            k: {"episodes": 20, "episode_results": rows}
            for k in ("full_stats", "validation_a", "validation_b")
        }
    }


def test_delivery_gate_stays_strict_but_moderate_forcing_does_not_stop_learning():
    g = run.ContinuingGuard(GuardrailConfig(stop_on_regression=False))
    g.consider(suite(), "old")
    for i in range(4):
        result = g.consider(suite(force=3), str(i))
        assert not result["eligible"] and not result["stop"]
    assert g.best["checkpoint"] == "old"


def test_three_severe_force_measurements_stop_and_recovery_resets_streak():
    g = run.ContinuingGuard(GuardrailConfig(stop_on_regression=False))
    assert not g.consider(suite(force=15), "a")["stop"]
    assert not g.consider(suite(force=15), "b")["stop"]
    assert g.consider(suite(), "recovered")["severe_streak"] == 0
    for i in range(3):
        assert g.consider(suite(force=15), str(i))["stop"] == (i == 2)


def test_sustained_strength_loss_stops_but_truncation_is_immediate():
    g = run.ContinuingGuard(GuardrailConfig(stop_on_regression=False))
    g.consider(suite(wins=15), "peak")
    for i in range(3):
        assert g.consider(suite(wins=5), str(i))["stop"] == (i == 2)
    h = run.ContinuingGuard(GuardrailConfig(stop_on_regression=False))
    assert h.consider(suite(truncated=True), "cut")["stop"]


def test_supervisor_dispatches_to_new_continuation_script(tmp_path):
    child = Mock(pid=1)
    child.poll.return_value = 0
    with patch.object(run.recovery.subprocess, "Popen", return_value=child) as launch:
        run.recovery.batch(tmp_path, float("inf"))
    assert launch.call_count == 3
    assert Path(launch.call_args.args[0][2]) == SCRIPT
    child.terminate.assert_not_called()
