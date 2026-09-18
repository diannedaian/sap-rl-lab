import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_checkpoint_selection")


def fixture_evidence(audit):
    source = importlib.import_module("test_midgame_evidence_audit").result()
    source.update(episodes=2, league_sha256="fixture")
    count = importlib.import_module("audit_midgame_confirmation").exact_family_counts(source, 2, 10)
    families = {name: deepcopy(source) for name in audit.comparison.GENERATORS}
    counts = {name: deepcopy(count) for name in families}
    results = {
        str(seed): {
            selector: {"families": deepcopy(families)} for selector in audit.comparison.SELECTORS
        }
        for seed in audit.comparison.SEEDS
    }
    totals = {
        str(seed): {selector: deepcopy(counts) for selector in audit.comparison.SELECTORS}
        for seed in audit.comparison.SEEDS
    }
    return results, totals, dict.fromkeys(results, 0)


def test_reconstruct_comparison_uses_audited_integer_counts(audit):
    results, counts, empty = fixture_evidence(audit)
    # Corrupt a copied aggregate; the reconstructed learned mean still uses counts.
    for pair in results.values():
        for result in pair.values():
            for family in result["families"].values():
                family["success_rate"] = 0.99
    actual = audit.comparison_from_rows(results, counts, empty)
    assert len(actual["pairs"]) == 5
    for pair in actual["pairs"].values():
        assert pair["win_first"]["learned_success"] == 0.5
        assert pair["reliability_first"]["learned_success"] == 0.5
    assert actual["mean_learned_difference"] == 0
    assert actual["seed_bootstrap_95_percent"] == [0, 0]
    assert not actual["practical_screen_passed"]  # Synthetic rows have forcing/empty failures.
    assert not actual["recipe_confirmed"]


def test_missing_seed_remains_in_audit_and_suppresses_survivor_average(audit):
    results, counts, empty = fixture_evidence(audit)
    seed = str(audit.comparison.SEEDS[0])
    results[seed]["reliability_first"] = counts[seed]["reliability_first"] = None
    actual = audit.comparison_from_rows(results, counts, empty)
    assert len(actual["pairs"]) == 5
    assert actual["missing_reliable_seeds"] == [seed]
    assert actual["mean_learned_difference"] is None
    assert actual["seed_bootstrap_95_percent"] is None
    assert not any(actual["gates"].values())


def test_report_audit_rejects_unearned_approval_and_changed_result(audit):
    results, counts, empty = fixture_evidence(audit)
    expected = audit.comparison_from_rows(results, counts, empty)
    audit.verify_summary(expected, expected)
    for key, wrong in [
        ("mean_learned_difference", 0.1),
        ("recipe_confirmed", True),
        ("missing_reliable_seeds", ["hidden"]),
        ("new_training_decisions", 1),
    ]:
        with pytest.raises(ValueError, match=key):
            audit.verify_summary({**expected, key: wrong}, expected)


def test_result_must_bind_both_model_and_frozen_selection(audit):
    expected = {"model_sha256": "model", "selection_sha256": "selection"}
    item = {"sha256": "model"}
    audit.result_binding(expected, expected, item, "selection")
    with pytest.raises(ValueError, match="binding"):
        audit.result_binding({**expected, "model_sha256": "different"}, expected, item, "selection")
    with pytest.raises(ValueError, match="binding"):
        audit.result_binding(expected, {**expected, "selection_sha256": "new"}, item, "selection")


def test_audit_cannot_open_results_before_selection_frozen(audit, monkeypatch, tmp_path):
    def no_selection(_):
        raise FileNotFoundError("selection.json")

    monkeypatch.setattr(audit.comparison, "frozen", no_selection)
    monkeypatch.setattr(audit, "read", lambda *args: pytest.fail("Must not open test results"))
    with pytest.raises(FileNotFoundError, match="selection.json"):
        audit.run(tmp_path, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_periodic_replay_gets_exact_adjacent_manifest_without_changing_original(audit, tmp_path):
    parent, output = tmp_path / "primary", tmp_path / "audit"
    checkpoint = parent / "control-seed123" / "checkpoints" / "periodic.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"exact selected weights")
    source_manifest = checkpoint.parent.parent / "run_manifest.json"
    source_manifest.write_text('{"config": {"action_cost": 0.005}}')
    item = {"path": str(checkpoint), "sha256": audit.file_digest(checkpoint), "seed": 123}
    inputs = audit.prepare_replay_inputs(output, parent, {"models": {"chosen": item}})
    copied = inputs["chosen"]
    assert audit.file_digest(copied["path"]) == audit.file_digest(checkpoint)
    assert Path(copied["path"]).parent / "run_manifest.json" == Path(copied["manifest_path"])
    assert audit.read(copied["manifest_path"])["config"]["action_cost"] == 0.005
    assert audit.file_digest(copied["manifest_path"]) == audit.file_digest(source_manifest)
    assert not (checkpoint.parent / "run_manifest.json").exists()
    assert checkpoint.read_bytes() == b"exact selected weights"
    with pytest.raises(FileExistsError):
        audit.prepare_replay_inputs(output, parent, {"models": {"chosen": item}})


def test_replay_input_copy_rejects_unselected_weights(audit, tmp_path):
    model = tmp_path / "model.zip"
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source model changed"):
        audit.prepare_replay_inputs(
            tmp_path / "out",
            tmp_path,
            {"models": {"chosen": {"path": str(model), "sha256": "old", "seed": 1}}},
        )
