import importlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_stability_confirmation")


def test_history_audit_rejects_later_equal_checkpoint(audit, monkeypatch):
    monkeypatch.setattr(audit, "midgame_checkpoint_score", lambda row: tuple(row["score"]))
    first = {"score": [0.5, 0.1], "selected": True, "evaluation_file": "first"}
    later = {"score": [0.5, 0.1], "selected": False, "evaluation_file": "last"}
    assert audit.history_selection([first, later]) == first
    with pytest.raises(ValueError, match="earliest-best"):
        audit.history_selection([first, {**later, "selected": True}])
    with pytest.raises(ValueError, match="Missing"):
        audit.history_selection([])


def test_suite_audit_recounts_empty_losses_and_checks_all_inputs(audit, tmp_path):
    fixture = importlib.import_module("test_midgame_evidence_audit")
    data = fixture.result()
    pool = tmp_path / "pool.json"
    pool.write_text("{}")
    contract = {
        "schema_version": 1,
        "catalog_id": "v5",
        "catalog_sha256": "c",
        "game_config": {"max_shop_tier": 3},
    }
    data.update(
        league_sha256=audit.file_digest(pool), deterministic=True, environment_contract=contract
    )
    suite = {
        "families": {"a": data},
        "episodes_per_family": 2,
        "seed_start": 10,
        "forced_end_turns": 2,
        "forced_end_turn_rate": 2 / 15,
        **{
            metric: data[metric]
            for metric in (
                "success_rate",
                "mean_return",
                "mean_wins",
                "truncation_rate",
                "mean_episode_actions",
                "forced_episode_rate",
                "success_without_forcing_rate",
            )
        },
    }
    counts = audit.recount_suite(suite, {"a": pool}, 2, 10, contract)
    assert counts["a"]["no_purchase_zero_win_losses"] == 1
    assert counts["a"]["forced_episodes"] == 1
    with pytest.raises(ValueError, match="Missing evaluation family"):
        audit.recount_suite(suite, {"a": pool, "b": pool}, 2, 10, contract)
    changed = deepcopy(suite)
    changed["families"]["a"]["deterministic"] = False
    with pytest.raises(ValueError, match="stochastic"):
        audit.recount_suite(changed, {"a": pool}, 2, 10, contract)
    with pytest.raises(ValueError, match="Missing, reordered"):
        audit.recount_suite(suite, {"a": pool}, 3, 10, contract)
    with pytest.raises(ValueError, match="macro average"):
        audit.recount_suite({**suite, "success_rate": 1}, {"a": pool}, 2, 10, contract)
    with pytest.raises(ValueError, match="forcing count"):
        audit.recount_suite({**suite, "forced_end_turns": 1}, {"a": pool}, 2, 10, contract)
    pool.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="pool changed"):
        audit.recount_suite(suite, {"a": pool}, 2, 10, contract)


def test_model_settings_must_match_frozen_config_and_manifest(audit):
    config = audit.asdict(audit.TrainingConfig(entropy_coefficient=0.01))
    manifest = {
        "config": audit.json.loads(audit.json.dumps(config)),
        "initialization": "from_scratch",
        "initial_model_sha256": None,
        "training_mixture_sha256": {},
        "validation_suite_sha256": {},
    }
    model = SimpleNamespace(
        ent_coef=0.01,
        seed=7,
        n_envs=4,
        n_steps=512,
        batch_size=256,
        gamma=1.0,
        gae_lambda=0.95,
        learning_rate=3e-4,
    )
    assert audit.check_training_configuration(manifest, config, model)["ent_coef"] == 0.01
    model.ent_coef = 0.0
    with pytest.raises(ValueError, match="Saved PPO setting.*ent_coef"):
        audit.check_training_configuration(manifest, config, model)
    model.ent_coef = 0.01
    with pytest.raises(ValueError, match="configuration differs"):
        audit.check_training_configuration(manifest, {**config, "action_cost": 0.005}, model)
    manifest["initialization"] = "policy_weights_only_fresh_optimizer"
    with pytest.raises(ValueError, match="from scratch"):
        audit.check_training_configuration(manifest, config, model)
