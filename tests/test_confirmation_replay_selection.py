"""Diagnostic sample selection cannot silently change the frozen game outcomes."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "inspect_confirmation_delivery",
    Path(__file__).resolve().parents[1] / "scripts/inspect_confirmation_delivery.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(seed, success=False, forced=0, truncated=False):
    return {
        "seed": seed,
        "success": success,
        "wins": 10 if success else 4,
        "return": 8.0 if success else -1.0,
        "actions": 20,
        "action_counts": {"end_turn": 20},
        "forced_end_turns": forced,
        "truncated": truncated,
    }


def test_earliest_cases_are_bounded_sorted_and_outcome_stratified():
    rows = [row(i, success=i >= 10) for i in range(20)]
    chosen = module.choose_rows(list(reversed(rows)))
    assert [r["seed"] for r in chosen] == [0, 1, 2, 3, 4, 10, 11, 12]


def test_forced_cases_included_even_if_outside_earliest_successes():
    rows = [row(i, success=True, forced=int(i >= 10)) for i in range(20)]
    assert [r["seed"] for r in module.choose_rows(rows)] == [0, 1, 2, 10, 11, 12, 13, 14]


def test_overlap_deduplicates_and_empty_group_does_not_invent_failures():
    rows = [row(i, forced=1, truncated=True) for i in range(7)]
    assert len(module.choose_rows(rows)) == 5
    assert module.choose_rows([]) == []


def test_duplicate_seeds_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        module.choose_rows([row(1), row(1, success=True)])


def replay(expected):
    actual = dict(expected)
    actual["raw_return"] = actual.pop("return")
    actual["forced_turns"] = actual.pop("forced_end_turns")
    actual["objective_return"] = -99  # shaped reward is intentionally not raw evaluation
    return actual


def test_exact_game_summary_allows_different_shaped_objective():
    expected = row(123)
    module.compare_replay(replay(expected), expected)


@pytest.mark.parametrize("field", ["actions", "raw_return", "forced_turns", "wins"])
def test_any_changed_game_summary_fails(field):
    expected = row(123)
    actual = replay(expected)
    actual[field] += 1
    with pytest.raises(ValueError, match=field):
        module.compare_replay(actual, expected)


def prepared_confirmation(tmp_path, monkeypatch):
    root = tmp_path / "confirmation"
    root.mkdir()
    model, league = root / "model.zip", root / "pool.json"
    model.write_bytes(b"trusted-test-model-placeholder")
    league.write_text("{}")

    def save(path, data):
        path.write_text(json.dumps(data))

    protocol = {"paths": {split: {"family": str(league)} for split in ("test", "challenge")}}
    save(root / "protocol.json", protocol)
    model_hash = module.file_digest(model)
    save(
        root / "selection.json",
        {
            "protocol_sha256": module.file_digest(root / "protocol.json"),
            "delivery": "candidate",
            "models": {"candidate": {"path": str(model), "sha256": model_hash}},
        },
    )
    selection_hash = module.file_digest(root / "selection.json")
    save(root / "summary.json", {"selection_sha256": selection_hash})
    save(
        root / "evaluation-candidate-started.json",
        {
            "selection_sha256": selection_hash,
            "model_sha256": model_hash,
        },
    )
    save(
        root / "evaluation-candidate.json",
        {
            split: {
                "families": {
                    "family": {
                        "league_sha256": module.file_digest(league),
                        "deterministic": True,
                        "episode_results": [row(123), row(124, success=True)],
                    }
                }
            }
            for split in ("test", "challenge")
        },
    )
    (root / "post-training").mkdir()
    save(root / "post-training/complete.json", {"stages_complete": True})
    monkeypatch.setattr(module, "verify_inputs", lambda *args: None)
    return root


def fake_inspector(monkeypatch, mismatch=False):
    from sap_rl_lab import expanded_diagnostics

    calls = []

    def inspect(model, league, directory, episodes, seed):
        assert episodes == 1
        calls.append(seed)
        directory.mkdir()
        summary = replay(row(seed, success=seed == 124))
        if mismatch:
            summary["actions"] += 1
        (directory / "summary.json").write_text(json.dumps({"episodes": [summary]}))

    monkeypatch.setattr(expanded_diagnostics, "inspect", inspect)
    return calls


def test_completed_pipeline_reconstruction_still_requires_manual_inspection(tmp_path, monkeypatch):
    root = prepared_confirmation(tmp_path, monkeypatch)
    calls = fake_inspector(monkeypatch)
    output = tmp_path / "replays"
    module.run(root, output)
    assert calls == [123, 124, 123, 124]
    complete = json.loads((output / "complete.json").read_text())
    assert len(complete["cases"]) == 4
    assert complete["manual_inspection_done"] is False
    assert not (output / "failed.json").exists()
    with pytest.raises(FileExistsError):
        module.run(root, output)
    assert len(calls) == 4


def test_incomplete_evaluation_barrier_prevents_reconstruction(tmp_path, monkeypatch):
    root = prepared_confirmation(tmp_path, monkeypatch)
    (root / "post-training/complete.json").write_text('{"stages_complete": false}')
    calls = fake_inspector(monkeypatch)
    output = tmp_path / "replays"
    with pytest.raises(ValueError, match="must complete"):
        module.run(root, output)
    assert calls == [] and not output.exists()


def test_changed_checkpoint_prevents_reconstruction(tmp_path, monkeypatch):
    root = prepared_confirmation(tmp_path, monkeypatch)
    (root / "model.zip").write_bytes(b"changed")
    calls = fake_inspector(monkeypatch)
    with pytest.raises(ValueError, match="checkpoint changed"):
        module.run(root, tmp_path / "replays")
    assert calls == []


def test_replay_mismatch_records_failure_and_stops_before_other_cases(tmp_path, monkeypatch):
    root = prepared_confirmation(tmp_path, monkeypatch)
    calls = fake_inspector(monkeypatch, mismatch=True)
    output = tmp_path / "replays"
    with pytest.raises(ValueError, match="actions"):
        module.run(root, output)
    assert calls == [123]
    assert (output / "failed.json").exists()
    assert not (output / "complete.json").exists()
