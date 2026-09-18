import pytest

from sap_rl_lab import historical_confirmation as hc
from sap_rl_lab.expanded_confirmation import save_new


def test_fallback_counts_only_played_battles():
    result = hc.fallback_counts([{"battles": 2}, {"battles": 4}], [1, 2, 4])
    assert result == {
        "battles": 6,
        "fallback_battles": 1,
        "fallback_battle_rate": 1 / 6,
        "episodes_with_fallback": 1,
    }
    assert hc.fallback_counts([], [])["fallback_battle_rate"] == 0


def test_copy_is_write_once_and_preserves_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "copy"
    source.write_bytes(b"original checkpoint")
    digest = hc.copy_checked(source, target)
    assert digest == hc.file_digest(str(source)) == hc.file_digest(str(target))
    with pytest.raises(FileExistsError):
        hc.copy_checked(source, target)
    assert source.read_bytes() == b"original checkpoint"


def test_sealed_test_content_tamper_is_detected(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text("original")
    save_new(
        tmp_path / "sealed_tests.json",
        {
            "all_test_data_sha256": {str(path): hc.file_digest(str(path))},
            "generators": {},
        },
    )
    hc.check_tests(tmp_path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="test data changed"):
        hc.check_tests(tmp_path)


def test_pipeline_seals_tests_before_training_and_reloads_all_before_testing(monkeypatch, tmp_path):
    calls = []
    protocol = {
        "generator_configs": {"fresh_a": {}, "fresh_b": {}},
        "arms": {f"{arm}-seed{s}": {} for s in hc.SEEDS for arm in hc.ARMS},
    }
    monkeypatch.setattr(hc, "prepare", lambda p: calls.append("prepare"))
    monkeypatch.setattr(hc, "checked", lambda p: protocol)
    monkeypatch.setattr(hc, "seal_tests", lambda p: calls.append("seal"))
    monkeypatch.setattr(hc, "freeze_selection", lambda p: calls.append("freeze"))
    monkeypatch.setattr(hc, "summarize", lambda p: calls.append("summarize"))
    monkeypatch.setattr(hc, "batch", lambda p, stage, names: calls.append((stage, len(names))))
    hc.pipeline(tmp_path)
    assert calls == [
        "prepare",
        ("generator", 2),
        "seal",
        ("arm", 6),
        "freeze",
        ("reload", 6),
        ("test", 6),
        "summarize",
    ]
    assert (tmp_path / "pipeline_complete.json").exists()


def test_pipeline_failure_does_not_launch_later_stages(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(hc, "prepare", lambda p: None)
    monkeypatch.setattr(hc, "checked", lambda p: {"generator_configs": {"one": {}}, "arms": {}})

    def fail(p, stage, names):
        calls.append(stage)
        raise RuntimeError("numeric failure")

    monkeypatch.setattr(hc, "batch", fail)
    with pytest.raises(RuntimeError, match="numeric failure"):
        hc.pipeline(tmp_path)
    assert calls == ["generator"]
    assert (tmp_path / "pipeline_failed.json").exists()
    assert not (tmp_path / "pipeline_complete.json").exists()
