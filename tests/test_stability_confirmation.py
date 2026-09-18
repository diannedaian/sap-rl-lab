from copy import deepcopy

import pytest

from sap_rl_lab import exploration_pilot as ep
from sap_rl_lab import stability_confirmation as sc
from sap_rl_lab.expanded_confirmation import read, save_new


def good_pairs():
    return {
        str(seed): {
            "candidate": {
                "worst_family_no_purchase_loss_rate": 0,
                "worst_family_forced_episode_rate": 0.015,
                "worst_family_truncation_rate": 0,
            },
            "learned_difference": 0.01,
            "final_no_purchase_loss_rate": 0,
        }
        for seed in sc.SEEDS
    }


def test_confirmation_has_five_fresh_paired_seeds_and_two_new_generators():
    assert len(sc.SEEDS) == len(set(sc.SEEDS)) == 5
    assert len(sc.GENERATORS) == 2
    assert not set(sc.SEEDS) & set(ep.SEEDS)
    assert not set(sc.SEEDS) & set(sc.GENERATORS.values())
    assert 10 * sc.CANDIDATE_STEPS + 2 * sc.GENERATOR_STEPS == 92_274_688


def test_acceptance_allows_one_point_five_percent_not_old_one_percent_gate():
    pairs = good_pairs()
    assert all(sc.acceptance(pairs).values())
    pairs[str(sc.SEEDS[0])]["candidate"]["worst_family_forced_episode_rate"] = 0.02
    assert sc.acceptance(pairs)["forcing_controlled"]
    pairs[str(sc.SEEDS[0])]["candidate"]["worst_family_forced_episode_rate"] = 0.02001
    assert not sc.acceptance(pairs)["forcing_controlled"]


def test_acceptance_rejects_empty_losses_even_with_zero_forcing():
    pairs = good_pairs()
    failed = pairs[str(sc.SEEDS[0])]
    failed["candidate"]["worst_family_forced_episode_rate"] = 0
    failed["candidate"]["worst_family_no_purchase_loss_rate"] = 1
    assert not sc.acceptance(pairs)["no_purchase_losses_controlled"]
    failed["candidate"]["worst_family_no_purchase_loss_rate"] = 0
    failed["final_no_purchase_loss_rate"] = 1
    assert not sc.acceptance(pairs)["no_final_validation_empty_relapse"]


def test_acceptance_does_not_hide_bad_pairs_behind_one_big_gain():
    pairs = good_pairs()
    for row in pairs.values():
        row["learned_difference"] = -0.01
    pairs[str(sc.SEEDS[0])]["learned_difference"] = 0.5
    gates = sc.acceptance(pairs)
    assert gates["mean_learned_performance_not_lower"]
    assert not gates["at_least_four_nonnegative_pairs"]
    assert not all(sc.acceptance({}).values())


def test_generator_quality_does_not_admit_empty_or_truncated_opponents():
    rows = [{"wins": 5, "action_counts": {"buy_pet": 5}, "truncated": False} for _ in range(500)]
    assert sc.generator_quality(rows)["usable"]
    with pytest.raises(ValueError, match="500"):
        sc.generator_quality(rows[:100])
    changed = deepcopy(rows)
    changed[0]["truncated"] = True
    assert not sc.generator_quality(changed)["usable"]
    for row in changed:
        row.update(wins=0, action_counts={"end_turn": 6}, truncated=False)
    assert not sc.generator_quality(changed)["usable"]


def test_all_reloads_barrier_checks_selection_and_model_hashes(tmp_path):
    selection = {"models": {"a": {"sha256": "model-a"}, "b": {"sha256": "model-b"}}}
    save_new(tmp_path / "selection.json", selection)
    digest = sc.file_digest(tmp_path / "selection.json")
    save_new(
        tmp_path / "reload-a.json",
        {"all_validation_rows_exact": True, "model_sha256": "model-a", "selection_sha256": digest},
    )
    with pytest.raises(FileNotFoundError):
        sc.require_all_reloads(tmp_path, selection)
    save_new(
        tmp_path / "reload-b.json",
        {
            "all_validation_rows_exact": True,
            "model_sha256": "model-b",
            "selection_sha256": "wrong-selection",
        },
    )
    with pytest.raises(ValueError, match="All frozen"):
        sc.require_all_reloads(tmp_path, selection)


def test_pipeline_freezes_and_reloads_all_ten_before_any_test(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sc, "prepare", lambda p, d: calls.append("prepare"))
    monkeypatch.setattr(sc, "seal", lambda p: calls.append("seal"))
    monkeypatch.setattr(sc, "freeze", lambda p: calls.append("freeze"))
    monkeypatch.setattr(sc, "verify", lambda p: {"arms": dict.fromkeys(range(10))})
    monkeypatch.setattr(sc, "batch", lambda p, stage, names: calls.append((stage, len(names))))
    monkeypatch.setattr(sc, "summarize", lambda p: calls.append("summarize"))
    sc.pipeline(tmp_path, tmp_path / "decision.json")
    assert calls == [
        "prepare",
        ("generator", 2),
        "seal",
        ("arm", 10),
        "freeze",
        ("reload", 10),
        ("test", 10),
        "summarize",
    ]
    assert read(tmp_path / "pipeline_complete.json")["human_readable_report_pending"]


def test_confirmation_failure_does_not_retry(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sc, "prepare", lambda p, d: None)

    def fail(p, stage, names):
        calls.append(stage)
        raise RuntimeError("worker failed")

    monkeypatch.setattr(sc, "batch", fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        sc.pipeline(tmp_path, tmp_path / "decision.json")
    assert calls == ["generator"]
    assert not read(tmp_path / "pipeline_failed.json")["automatic_retry"]


def test_confirmation_batch_has_at_most_two_owned_children(monkeypatch, tmp_path):
    children = []

    class Child:
        def __init__(self, cmd, cwd):
            assert sum(c.returncode is None for c in children) < 2
            self.returncode = None
            children.append(self)

        def poll(self):
            self.returncode = 0
            return 0

    monkeypatch.setattr(sc.subprocess, "Popen", Child)
    sc.batch(tmp_path, "arm", list(map(str, range(10))))
    assert len(children) == 10


@pytest.fixture
def reviewed_sensitivity(monkeypatch, tmp_path):
    prior = tmp_path / "runs/exploration-sensitivity-v1"
    prior.mkdir(parents=True)
    train, validation = prior / "train.json", prior / "validation.json"
    save_new(train, {"source": "training"})
    save_new(validation, {"source": "validation"})
    save_new(
        prior / "protocol.json",
        {
            "source_files_sha256": {},
            "data_sha256": {},
            "paths": {
                "train": {sc.FAMILIES[0]: str(train)},
                "validation": {"val": str(validation)},
            },
        },
    )
    save_new(
        prior / "pilot_summary.json",
        {
            "all_six_budgets_complete": True,
            "all_selected_validation_reloads_exact": True,
            "paired_initial_weights_and_rows_exact": True,
        },
    )
    audit = tmp_path / "audit"
    audit.mkdir()
    save_new(audit / "summary.json", {"all_curves_recounted": True, "all_replays_exact": True})
    save_new(audit / "plan.json", {"protocol_sha256": sc.file_digest(prior / "protocol.json")})
    decision = tmp_path / "decision.json"
    save_new(
        decision,
        {
            "candidate_entropy": 0.01,
            "rationale": "Reviewed all paired curves and replays.",
            "pilot_summary_sha256": sc.file_digest(prior / "pilot_summary.json"),
            "replay_audit": str(audit / "summary.json"),
            "replay_audit_sha256": sc.file_digest(audit / "summary.json"),
        },
    )
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    monkeypatch.setattr(sc, "source_archive", lambda output: {})
    monkeypatch.setattr(
        sc,
        "check_parent",
        lambda output: {"roles": {"train": {"train_policy": {}}, "validation": {"val_policy": {}}}},
    )
    calls = []

    class League:
        def save(self, path):
            save_new(path, {"scripted": True})

    def build(family, episodes, seed, **kwargs):
        calls.append((family, episodes, seed))
        return League()

    monkeypatch.setattr(sc, "build_scripted_league", build)
    return prior, audit, decision, calls


def test_prepare_copies_reviewed_inputs_and_fixes_new_split_budgets(reviewed_sensitivity, tmp_path):
    prior, _, decision, calls = reviewed_sensitivity
    output = tmp_path / "confirmation"
    sc.prepare(output, decision)
    design = sc.checked(output)
    assert design["candidate_entropy"] == 0.01
    assert design["candidate_steps"] == sc.CANDIDATE_STEPS
    assert design["total_training_decisions"] == 92_274_688
    config = design["base_config"]
    assert config["validation_episodes"] == 200
    assert config["validation_seed"] == sc.VALIDATION_SEED
    assert config["action_cost"] == 0.005
    assert config["initialize_from"] == ""
    assert read(design["paths"]["validation"]["val"]) == read(prior / "validation.json")
    assert len(calls) == 3 and all(c[1] == 500 for c in calls)
    assert len({c[2] for c in calls}) == 3
    for name, config in design["generator_configs"].items():
        assert config["timesteps"] == sc.GENERATOR_STEPS
        assert config["entropy_coefficient"] == 0.01
        assert config["seed"] == sc.GENERATORS[name]
        assert config["initialize_from"] == ""
    with pytest.raises(FileExistsError):
        sc.prepare(output, decision)


def test_prepare_rejects_a_replay_audit_for_another_protocol(reviewed_sensitivity, tmp_path):
    _, audit, decision, _ = reviewed_sensitivity
    # Rewrite only this temporary test fixture, not actual experiment evidence.
    (audit / "plan.json").write_text('{"protocol_sha256": "different-experiment"}')
    with pytest.raises(ValueError, match="different pilot"):
        sc.prepare(tmp_path / "confirmation", decision)
    assert not (tmp_path / "confirmation").exists()


def test_checked_rejects_changed_review_decision(reviewed_sensitivity, tmp_path):
    _, _, decision, _ = reviewed_sensitivity
    output = tmp_path / "confirmation"
    sc.prepare(output, decision)
    (output / "candidate_decision.json").write_text("{}")
    with pytest.raises(ValueError, match="Candidate decision changed"):
        sc.checked(output)
