import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "finish_tier4", SCRIPTS / "finish_tier4_evaluation.py"
)
finish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finish)


def test_missing_selection_stays_missing_not_final_fallback():
    models = {}
    assert finish.bind_selection(3, ["initial", "final"], None, models) is None
    assert models == {"3-initial": "initial", "3-final": "final"}


@pytest.mark.parametrize(
    "selected,expected", [("initial", "3-initial"), ("final", "3-final"), ("middle", "3-selected")]
)
def test_selected_aliases_deduplicate_without_test_selection(selected, expected):
    models = {}
    actual = finish.bind_selection(
        3, ["initial", "middle", "final"], {"checkpoint": selected}, models
    )
    assert actual == expected
    assert len(models) == (3 if selected == "middle" else 2)


def test_selection_must_belong_to_frozen_history():
    with pytest.raises(ValueError, match="missing"):
        finish.bind_selection(3, ["initial", "final"], {"checkpoint": "unlisted"}, {})


def test_exact_reload_barrier(tmp_path):
    p = {"models": {"a": {"path": "a"}, "b": {"path": "b"}}}
    finish.write_new(tmp_path / "reload-a.json", {"checkpoint": {"path": "a"}, "exact_rows": 1000})
    with pytest.raises(FileNotFoundError):
        finish.require_reloads(tmp_path, p)
    finish.write_new(tmp_path / "reload-b.json", {"checkpoint": {"path": "b"}, "exact_rows": 999})
    with pytest.raises(ValueError, match="exact reload"):
        finish.require_reloads(tmp_path, p)


def test_expired_budget_launches_nothing(tmp_path, monkeypatch):
    launch = Mock()
    monkeypatch.setattr(finish.subprocess, "Popen", launch)
    with pytest.raises(TimeoutError):
        finish.batch(tmp_path, "test", ["a"], 0)
    launch.assert_not_called()


def test_failed_worker_keeps_error_and_closes_log(tmp_path, monkeypatch):
    process = Mock(pid=1)
    process.poll.return_value = 1
    launch = Mock(return_value=process)
    monkeypatch.setattr(finish.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="exit=1"):
        finish.batch(tmp_path, "reload", ["a"], float("inf"))
    assert launch.call_args.kwargs["stdout"].closed
    process.terminate.assert_not_called()
