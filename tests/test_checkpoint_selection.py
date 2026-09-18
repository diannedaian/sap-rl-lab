import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def selection(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_checkpoint_selection")


def candidate(name, step, wins, forcing, empty=0, truncation=0):
    return {
        "id": name,
        "timesteps": step,
        "score": [wins, wins, -forcing, wins],
        "metrics": {
            "worst_family_forced_episode_rate": forcing,
            "worst_family_no_purchase_loss_rate": empty,
            "worst_family_truncation_rate": truncation,
        },
    }


def test_same_grid_exposes_one_win_versus_large_forcing_tradeoff(selection):
    good = candidate("earlier", 5_242_880, 716 / 800, 0.02)
    bad = candidate("later", 8_388_608, 717 / 800, 0.65)
    assert selection.choose([good, bad]) == {"win_first": "later", "reliability_first": "earlier"}


def test_tie_uses_actual_steps_then_stable_id_not_input_order(selection):
    rows = [
        candidate("z", 20, 0.8, 0.01),
        candidate("b", 10, 0.8, 0.01),
        candidate("a", 10, 0.8, 0.01),
    ]
    assert selection.choose(rows) == dict.fromkeys(selection.SELECTORS, "a")
    assert selection.choose(rows[::-1]) == selection.choose(rows)


@pytest.mark.parametrize(
    "forcing,empty,trunc,expected",
    [
        (0.02, 0.01, 0, True),
        (0.02001, 0, 0, False),
        (0, 0.01001, 0, False),
        (0, 0, 0.001, False),
    ],
)
def test_each_reliability_gate_and_exact_boundary(selection, forcing, empty, trunc, expected):
    assert selection.eligible(candidate("x", 10, 0.8, forcing, empty, trunc)["metrics"]) is expected


def test_missing_eligible_model_is_explicit_failure_without_fallback(selection):
    bad = candidate("bad", 10, 0.9, 0.1)
    assert selection.choose([bad]) == {"win_first": "bad", "reliability_first": None}
    with pytest.raises(ValueError, match="Missing or duplicate"):
        selection.choose([])
    with pytest.raises(ValueError, match="Missing or duplicate"):
        selection.choose([bad, bad])


def test_all_five_required_no_survivor_only_pass(selection):
    pairs = {
        str(s): {
            "reliability_first": candidate("good", 1, 0.8, 0)["metrics"],
            "learned_difference": 0.01,
            "final_no_purchase_loss_rate": 0,
        }
        for s in selection.SEEDS
    }
    assert all(selection.acceptance(pairs).values())
    pairs[str(selection.SEEDS[0])]["reliability_first"] = None
    pairs[str(selection.SEEDS[0])]["learned_difference"] = None
    assert not any(selection.acceptance(pairs).values())
    del pairs[str(selection.SEEDS[0])]
    assert not any(selection.acceptance(pairs).values())


def test_better_mean_does_not_hide_three_negative_pairs(selection):
    pairs = {
        str(s): {
            "reliability_first": candidate("good", 1, 0.8, 0)["metrics"],
            "learned_difference": d,
            "final_no_purchase_loss_rate": 0,
        }
        for s, d in zip(selection.SEEDS, [0.1, 0.1, -0.01, -0.01, -0.01])
    }
    gates = selection.acceptance(pairs)
    assert gates["mean_learned_performance_not_lower"]
    assert not gates["at_least_four_nonnegative_pairs"]
    pairs[str(selection.SEEDS[0])]["final_no_purchase_loss_rate"] = 0.015
    assert not selection.acceptance(pairs)["no_final_empty_relapse"]


def test_registration_rejects_even_test_start_marker_without_reading_it(selection, tmp_path):
    selection.primary_not_tested(tmp_path)
    (tmp_path / "test-control-seed201101-started.json").write_text("NOT JSON")
    with pytest.raises(ValueError, match="no primary tests"):
        selection.primary_not_tested(tmp_path)


def test_test_inference_requires_every_reload_before_loading_any_model(
    selection, monkeypatch, tmp_path
):
    frozen = {"models": {"one": {"sha256": "one"}, "two": {"sha256": "two"}}}
    selection.save_new(tmp_path / "selection.json", frozen)
    selection.save_new(
        tmp_path / "reload-one.json",
        {
            "all_validation_rows_exact": True,
            "model_sha256": "one",
            "selection_sha256": selection.file_digest(tmp_path / "selection.json"),
        },
    )
    monkeypatch.setattr(selection, "frozen", lambda _: (tmp_path, {}, frozen))
    monkeypatch.setattr(
        selection, "load", lambda *a: pytest.fail("Must not load before all reloads")
    )
    with pytest.raises(FileNotFoundError):
        selection.inference(tmp_path, "one", "test")
    assert not list(tmp_path.glob("test-*.json"))
    selection.save_new(
        tmp_path / "reload-two.json",
        {
            "all_validation_rows_exact": True,
            "model_sha256": "WRONG",
            "selection_sha256": selection.file_digest(tmp_path / "selection.json"),
        },
    )
    with pytest.raises(ValueError, match="All frozen selections"):
        selection.require_reloads(tmp_path, frozen)


def test_readonly_pipeline_refuses_incomplete_or_failed_parent(selection, tmp_path):
    with pytest.raises(FileNotFoundError):
        selection.require_parent_complete(tmp_path)
    selection.save_new(
        tmp_path / "pipeline_complete.json", {"training_and_evaluation_complete": True}
    )
    selection.require_parent_complete(tmp_path)
    selection.save_new(tmp_path / "pipeline_failed.json", {"error": "failed"})
    with pytest.raises(ValueError, match="failed"):
        selection.require_parent_complete(tmp_path)


def test_frozen_choice_tampering_is_rejected(selection, monkeypatch, tmp_path):
    monkeypatch.setattr(selection, "SEEDS", (1,))
    a, b = candidate("a", 10, 0.8, 0.01), candidate("b", 20, 0.9, 0.8)
    selection.save_new(tmp_path / "inventory.json", {})
    selection.save_new(tmp_path / "validated-1.json", {"candidates": [a, b]})
    frozen = {
        "pairs": {"1": {"win_first": "b", "reliability_first": "b"}},
        "models": {},
        "inventory_sha256": selection.file_digest(tmp_path / "inventory.json"),
        "validation_records_sha256": {
            str(tmp_path / "validated-1.json"): selection.file_digest(tmp_path / "validated-1.json")
        },
    }
    selection.save_new(tmp_path / "selection.json", deepcopy(frozen))
    monkeypatch.setattr(selection, "indexed", lambda _: (tmp_path, {}, {}))
    with pytest.raises(ValueError, match="Frozen choices differ"):
        selection.frozen(tmp_path)


def test_registered_helper_changes_are_rejected(selection, monkeypatch, tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("original")
    selection.save_new(
        tmp_path / "registration.json",
        {
            "parent_protocol_sha256": selection.PARENT_SHA,
            "source_sha256": {str(helper): selection.file_digest(helper)},
        },
    )
    helper.write_text("changed")
    with pytest.raises(ValueError, match="Registered code or plan changed"):
        selection.checked(tmp_path)


def test_saved_steps_not_filename_or_manifest_steps(selection, monkeypatch, tmp_path):
    from types import SimpleNamespace

    model = tmp_path / "model.zip"
    model.write_bytes(b"fixture")
    item = {"path": str(model), "sha256": selection.file_digest(model), "timesteps": 838856}
    monkeypatch.setattr(selection, "metadata", lambda _: SimpleNamespace(num_timesteps=838857))
    with pytest.raises(ValueError, match="Actual saved model steps differ"):
        selection.check_model(item, tmp_path, {}, "control-seed1")


def test_missing_selector_summary_keeps_all_seeds_and_no_survivor_average(
    selection, monkeypatch, tmp_path
):
    """Synthetic inference summaries only; no training or actual evaluation workload."""
    parent = tmp_path / "parent"
    parent.mkdir()
    pairs, models = {}, {}
    family = importlib.import_module("test_midgame_evidence_audit").result()
    family.update(episodes=2, league_sha256="fixture")
    data = {"families": {name: deepcopy(family) for name in selection.GENERATORS}}
    for i, seed in enumerate(selection.SEEDS):
        name = str(seed)
        pairs[name] = {"win_first": name, "reliability_first": name if i else None}
        models[name] = {"sha256": name}
        folder = parent / f"control-seed{seed}"
        folder.mkdir()
        selection.save_new(folder / "validation_history.json", [{"evaluation_file": "final.json"}])
        selection.save_new(folder / "final.json", data)
    frozen = {"pairs": pairs, "models": models}
    selection.save_new(tmp_path / "selection.json", frozen)
    digest = selection.file_digest(tmp_path / "selection.json")
    for name in models:
        selection.save_new(
            tmp_path / f"test-{name}.json",
            {**deepcopy(data), "model_sha256": name, "selection_sha256": digest},
        )
    protocol = {
        "paths": {"test": dict.fromkeys(selection.GENERATORS, "fixture")},
        "environment_contract": {},
    }
    monkeypatch.setattr(selection, "frozen", lambda _: (parent, protocol, frozen))
    monkeypatch.setattr(selection, "require_reloads", lambda *args: None)
    monkeypatch.setattr(selection, "recount_suite", lambda *args: {})
    selection.summarize(tmp_path)
    summary = selection.read(tmp_path / "comparison_summary.json")
    assert len(summary["pairs"]) == 5
    assert summary["missing_reliable_seeds"] == [str(selection.SEEDS[0])]
    assert summary["mean_learned_difference"] is None
    assert summary["seed_bootstrap_95_percent"] is None
    assert not summary["practical_screen_passed"]
    assert not summary["recipe_confirmed"]
