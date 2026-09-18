from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from sap_rl_lab import imitation
from sap_rl_lab import imitation_experiment as experiment
from sap_rl_lab.expanded_confirmation import compare_reload, read
from sap_rl_lab.midgame_confirmation import new_config
from sap_rl_lab.training import TrainingConfig, policy_digest, train


@pytest.fixture
def config():
    return TrainingConfig(
        **new_config(
            seed=902101,
            timesteps=32,
        )
    )


@pytest.fixture
def tiny_data(config, tmp_path):
    records = imitation.collect(config, ("midgame_stats",), 2, 991000, tmp_path / "data.npz")
    return imitation.load_data(tmp_path / "data.npz"), records


def test_collected_labels_legal_and_episode_boundaries(tiny_data):
    data, records = tiny_data
    assert data["masks"][np.arange(len(data["actions"])), data["actions"]].all()
    assert len(records["episodes"]) == 2
    assert records["episodes"][0]["end"] == records["episodes"][1]["start"]
    assert records["episodes"][-1]["end"] == records["rows"]
    assert records["action_counts"].get("buy_pet", 0) > 0
    assert records["action_counts"].get("toggle_freeze", 0) == 0
    assert not records["rewards_used_for_labels"]


def test_episode_split_rejects_leakage(tiny_data):
    data, _ = tiny_data
    with pytest.raises(ValueError, match="leakage"):
        imitation.validate_split(data, data)
    heldout = {k: v.copy() for k, v in data.items()}
    heldout["episode_ids"] += 1_000_000
    imitation.validate_split(data, heldout)


def test_bad_labels_and_nonfinite_data_rejected(tiny_data):
    data, _ = tiny_data
    bad = deepcopy(data)
    bad["masks"][0, bad["actions"][0]] = False
    with pytest.raises(ValueError, match="Illegal"):
        imitation.validate_data(bad)
    bad = deepcopy(data)
    bad["actions"][0] = -1
    with pytest.raises(ValueError, match="range"):
        imitation.validate_data(bad)
    bad = deepcopy(data)
    bad["obs__global"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        imitation.validate_data(bad)


def test_demonstration_collection_refuses_overwrite(config, tmp_path):
    path = tmp_path / "data.npz"
    imitation.collect(config, ("midgame_stats",), 1, 992000, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        imitation.collect(config, ("midgame_stats",), 1, 992000, path)
    assert path.read_bytes() == before


def test_actor_only_fit_preserves_critic_optimizer_and_initial_model(config, tiny_data, tmp_path):
    from sb3_contrib import MaskablePPO

    data, _ = tiny_data
    heldout = deepcopy(data)
    heldout["episode_ids"] += 1_000_000
    model = imitation.fresh_model(config)
    initial = policy_digest(model.policy)
    critic = {k: v.clone() for k, v in model.policy.state_dict().items() if "value_net" in k}
    result = imitation.fit(model, data, heldout, tmp_path / "fit", epochs=2, seed=3)
    assert result["history"][0]["policy_sha256"] == initial
    assert result["selected"]["epoch"] in (1, 2)
    assert result["training_example_visits"] == 2 * len(data["actions"])
    assert policy_digest(model.policy) != initial
    assert all(torch.equal(v, model.policy.state_dict()[k]) for k, v in critic.items())
    assert not model.policy.optimizer.state
    restored = MaskablePPO.load(result["selected"]["path"], device="cpu")
    assert policy_digest(restored.policy) == result["selected"]["policy_sha256"]
    with pytest.raises(FileExistsError):
        imitation.fit(model, data, heldout, tmp_path / "fit", epochs=1)
    model.get_env().close()


def test_new_shared_features_require_explicit_review(config):
    model = imitation.fresh_model(config)
    model.policy.register_parameter("unreviewed", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="Unreviewed"):
        imitation.actor_parameters(model.policy)
    model.get_env().close()


def passing_rows():
    return [
        {
            "initial_metrics": {"mean_wins": 0.5},
            "bc_metrics": {
                "mean_wins": 2,
                "worst_family_no_purchase_loss_rate": 0,
                "worst_family_forced_episode_rate": 0.1,
                "worst_family_truncation_rate": 0,
            },
        }
        for _ in range(3)
    ]


def test_gate_checks_closed_loop_not_imitation_accuracy():
    rows = passing_rows()
    assert all(experiment.bc_gate(rows).values())
    rows[0]["bc_metrics"]["worst_family_no_purchase_loss_rate"] = 1
    assert not experiment.bc_gate(rows)["nonempty_play"]
    rows = passing_rows()
    rows[2]["bc_metrics"]["worst_family_forced_episode_rate"] = 0.20001
    assert not experiment.bc_gate(rows)["not_mostly_forced"]
    rows = passing_rows()
    rows[1]["bc_metrics"]["mean_wins"] = 0.9
    assert not experiment.bc_gate(rows)["basic_strength"]
    with pytest.raises(ValueError, match="three"):
        experiment.bc_gate([])


def test_budget_and_pairing_are_not_reward_ablation():
    assert experiment.STEPS * len(experiment.SEEDS) * 2 == 50_331_648
    assert experiment.SEEDS == (17301, 204101, 204201)
    assert experiment.VAL_SEED != experiment.TEST_SEED


def test_exact_bc_warmstart_to_fresh_ppo_optimizer(config, tiny_data, tmp_path):
    from sb3_contrib import MaskablePPO

    data, _ = tiny_data
    heldout = deepcopy(data)
    heldout["episode_ids"] += 1_000_000
    model = imitation.fresh_model(config)
    result = imitation.fit(model, data, heldout, tmp_path / "bc", epochs=1)
    model.get_env().close()
    warm = result["selected"]
    train_config = replace(
        config,
        environments=1,
        rollout_steps=8,
        batch_size=8,
        output_dir=str(tmp_path / "ppo"),
        initialize_from=warm["path"],
        expected_initial_policy_sha256=warm["policy_sha256"],
    )
    final_path = train(train_config)
    manifest = read(tmp_path / "ppo/run_manifest.json")
    assert manifest["initial_policy_sha256"] == warm["policy_sha256"]
    assert manifest["input_migration"] == "exact_policy_weights"
    assert manifest["initialization"] == "policy_weights_only_fresh_optimizer"
    final = MaskablePPO.load(str(final_path), device="cpu")
    assert final.num_timesteps == 32
    assert final.sap_environment_contract == model.sap_environment_contract
    assert policy_digest(final.policy) != warm["policy_sha256"]


def test_random_initialization_matches_existing_trainer(config, tmp_path):
    from unittest.mock import patch

    from sap_rl_lab import evaluation
    from sap_rl_lab.opponents import build_scripted_league
    from sap_rl_lab.reward_cost_pilot import checkpointing_evaluator

    model = imitation.fresh_model(config)
    expected = policy_digest(model.policy)
    catalog, game = config.environment()
    pool = tmp_path / "league.json"
    build_scripted_league("midgame_stats", 1, 923400, catalog=catalog, config=game).save(pool)
    kwargs = {"episodes": 1, "seed": 923600}
    initial_eval = evaluation.evaluate_suite(model, {"stats": str(pool)}, **kwargs)
    model.get_env().close()
    config = replace(
        config,
        environments=1,
        rollout_steps=8,
        batch_size=8,
        output_dir=str(tmp_path / "ppo"),
        expected_initial_policy_sha256=expected,
        validation_leagues={"stats": str(pool)},
        validation_seed=923600,
        validation_episodes=1,
        evaluation_interval=16,
    )
    with patch.object(
        evaluation,
        "evaluate_suite",
        checkpointing_evaluator(evaluation.evaluate_suite, tmp_path / "ppo"),
    ):
        train(config)
    compare_reload(initial_eval, read(tmp_path / "ppo/validation_0_eval000.json"))
    history = read(tmp_path / "ppo/validation_history.json")
    assert len(history) == 4
    assert history[-1]["timesteps"] == history[-2]["timesteps"] == 32
    assert history[-1]["evaluation_file"] != history[-2]["evaluation_file"]
