import pytest

from sap_rl_lab import exploration_pilot as pilot
from sap_rl_lab import exploration_sensitivity as sensitivity
from sap_rl_lab.expanded_confirmation import read, save_new


def test_sensitivity_spends_only_reserved_supplementary_budget():
    assert sensitivity.STEPS == 2_097_152
    assert sensitivity.ENTROPIES == {"entropy0003": 0.003, "entropy001": 0.01}
    assert sensitivity.SEEDS == pilot.SEEDS
    assert sensitivity.STEPS * len(sensitivity.SEEDS) * len(sensitivity.ENTROPIES) == (
        2 * pilot.STEPS * len(pilot.SEEDS) * len(pilot.ENTROPIES)
    )
    # Both launch the existing frozen arm worker, which reads actual configurations
    # from the experiment protocol instead of hardcoding the pilot coefficient.
    assert sensitivity.batch is pilot.batch


def test_sensitivity_prepare_copies_inputs_and_preserves_other_settings(monkeypatch, tmp_path):
    prior = tmp_path / "runs/exploration-pilot-v1"
    prior.mkdir(parents=True)
    audit = tmp_path / "runs/exploration-pilot-audit-v1"
    audit.mkdir()
    train, validation = prior / "train.json", prior / "validation.json"
    save_new(train, {"source": "training"})
    save_new(validation, {"source": "validation"})
    config = pilot.new_config(timesteps=pilot.STEPS)
    save_new(
        prior / "protocol.json",
        {
            "source_files_sha256": {},
            "data_sha256": {},
            "base_config": config,
            "paths": {"train": {"a": str(train)}, "validation": {"b": str(validation)}},
        },
    )
    save_new(
        prior / "pilot_summary.json",
        {
            "all_six_budgets_complete": True,
            "all_selected_validation_reloads_exact": True,
        },
    )
    save_new(audit / "summary.json", {"all_curves_recounted": True, "all_replays_exact": True})
    monkeypatch.setattr(sensitivity, "ROOT", tmp_path)
    monkeypatch.setattr(sensitivity, "source_archive", lambda output: {})
    output = tmp_path / "new"
    sensitivity.prepare(output)
    p = read(output / "protocol.json")
    assert p["base_config"]["timesteps"] == sensitivity.STEPS
    assert p["base_config"]["action_cost"] == 0.005
    assert p["base_config"]["learning_rate"] == config["learning_rate"]
    assert p["base_config"]["initialize_from"] == ""
    assert read(p["paths"]["train"]["a"]) == read(train)
    assert read(p["paths"]["validation"]["b"]) == read(validation)
    for seed in sensitivity.SEEDS:
        assert p["arms"][f"entropy0003-seed{seed}"] == {
            "seed": seed,
            "entropy_coefficient": 0.003,
        }
        assert p["arms"][f"entropy001-seed{seed}"] == {
            "seed": seed,
            "entropy_coefficient": 0.01,
        }
    with pytest.raises(FileExistsError):
        sensitivity.prepare(output)
