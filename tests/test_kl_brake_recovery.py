"""Recovery chooses a common saved prefix by availability, not performance."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("recover_kl_brake", SCRIPTS / "recover_kl_brake.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_largest_common_saved_prefix_not_longest_completed_arm():
    full = list(range(0, 2097152 + 1, 262144))
    assert recovery.common_checkpoint_step([full] * 5 + [full[:-1]]) == 1835008


def test_all_arms_required():
    with pytest.raises(ValueError, match="six"):
        recovery.common_checkpoint_step([[0, 262144]] * 5)


def test_missing_middle_checkpoint_is_not_ignored():
    with pytest.raises(ValueError, match="contiguous"):
        recovery.common_checkpoint_step([[0, 524288]] * 6)


def test_initial_only_is_not_a_trained_comparison():
    with pytest.raises(ValueError, match="positive"):
        recovery.common_checkpoint_step([[0]] * 6)


def test_expired_recovery_does_not_start_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **kw: pytest.fail("no start"))
    with pytest.raises(TimeoutError):
        recovery.batch(tmp_path, "reload", ["model"], deadline=0)
