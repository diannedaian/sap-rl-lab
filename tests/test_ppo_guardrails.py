from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from sap_rl_lab import imitation
from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import read
from sap_rl_lab.midgame_confirmation import new_config
from sap_rl_lab.opponents import build_scripted_league
from sap_rl_lab.ppo_guardrails import (
    GuardrailConfig,
    ObservedMaskablePPO,
    ValidationGuard,
    isolated_evaluation_rng,
    train_guarded,
)
from sap_rl_lab.stable_masking import StableMaskableMultiInputPolicy
from sap_rl_lab.training import TrainingConfig, policy_digest


def suite(wins=8, successes=8, forced=0, empty=0, truncated=0):
    rows = [
        dict(
            wins=0 if i < empty else wins,
            success=i < successes,
            forced_end_turns=int(i < forced),
            truncated=i < truncated,
            action_counts={"buy_pet": int(i >= empty)},
            return_=1.0,
        )
        for i in range(10)
    ]
    for row in rows:
        row["return"] = row.pop("return_")
    return {"families": {"a": {"episodes": 10, "episode_results": rows}}}


def test_reliability_precedes_win_rate_and_stops_after_patience():
    guard = ValidationGuard(GuardrailConfig())
    assert guard.consider(suite(), "good")["selected"]
    assert not guard.consider(suite(successes=10, forced=10), "bad1")["stop"]
    decision = guard.consider(suite(successes=10, forced=10), "bad2")
    assert decision["stop"] and not decision["eligible"]
    assert guard.best["checkpoint"] == "good"


def test_initial_incompetence_does_not_select_or_stop_and_recovery_resets_streak():
    guard = ValidationGuard(GuardrailConfig())
    for _ in range(3):
        result = guard.consider(suite(wins=0, successes=0, empty=10), "empty")
        assert not result["stop"] and guard.best is None
    guard.consider(suite(), "good")
    guard.consider(suite(forced=9), "bad")
    recovered = guard.consider(suite(), "tie")
    assert recovered["bad_streak"] == 0 and not recovered["selected"]
    assert guard.best["checkpoint"] == "good"


def test_strength_drop_and_immediate_truncation_are_separate_guards():
    guard = ValidationGuard(GuardrailConfig())
    guard.consider(suite(successes=10), "good")
    first = guard.consider(suite(successes=8), "weak")
    assert first["eligible"] and first["regression_reasons"] == ["family_strength_regression"]
    assert guard.consider(suite(successes=8), "weak")["stop"]
    guard = ValidationGuard(GuardrailConfig(stop_on_regression=False))
    assert guard.consider(suite(truncated=1), "truncated")["stop"]


def test_missing_changed_families_or_nonfinite_values_fail_closed():
    guard = ValidationGuard(GuardrailConfig())
    with pytest.raises(ValueError, match="per-family"):
        guard.consider({}, "missing")
    guard.consider(suite(), "ok")
    changed = suite()
    changed["families"]["b"] = changed["families"].pop("a")
    with pytest.raises(ValueError, match="families changed"):
        guard.consider(changed, "changed")
    bad = suite()
    bad["families"]["a"]["episode_results"][0]["return"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        guard.consider(bad, "nan")


def test_macro_gain_does_not_erase_family_regression_reference():
    original = suite(successes=8)
    original["families"]["b"] = suite(successes=2)["families"]["a"]
    candidate = suite(successes=6)
    candidate["families"]["b"] = suite(successes=10)["families"]["a"]
    guard = ValidationGuard(GuardrailConfig())
    guard.consider(original, "original")
    assert not guard.consider(candidate, "candidate1")["selected"]
    assert guard.consider(candidate, "candidate2")["stop"]
    assert guard.best["checkpoint"] == "original"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_kl": 0},
        {"target_kl": np.nan},
        {"final_lr_fraction": 0},
        {"clip_range": 2},
        {"regression_patience": 0},
        {"max_seconds": np.inf},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        GuardrailConfig(**kwargs).validate()


def make_model(cls, *, target_kl=None):
    config = TrainingConfig(**new_config(seed=94201))
    env = DummyVecEnv([lambda: imitation.make_env(config)])
    model = cls(
        StableMaskableMultiInputPolicy,
        env,
        seed=94201,
        device="cpu",
        n_steps=32,
        batch_size=16,
        n_epochs=4,
        target_kl=target_kl,
        learning_rate=3e-4,
        gamma=1.0,
        ent_coef=0.0,
    )
    model.set_logger(configure(format_strings=[]))
    model.action_kinds = [
        env.envs[0].engine.codec.decode(i).kind.value for i in range(model.action_space.n)
    ]
    return model


def test_telemetry_does_not_change_ppo_rng_or_weights_and_includes_last_update(tmp_path):
    torch.set_num_threads(1)
    vanilla = make_model(MaskablePPO)
    try:
        vanilla.learn(128)
        expected = policy_digest(vanilla.policy)
    finally:
        vanilla.get_env().close()
    instrumented = make_model(ObservedMaskablePPO)
    updates = []
    instrumented.update_writer = updates.append
    try:
        instrumented.learn(128)
        assert policy_digest(instrumented.policy) == expected
        assert len(updates) == 4 and updates[-1]["timesteps"] == 128
        assert all(r["optimizer_steps"] == 8 and not r["kl_early_stopped"] for r in updates)
        assert all(r["exact_rollout_kl_mean"] >= 0 for r in updates)
        instrumented.save(tmp_path / "model.zip")
        reloaded = MaskablePPO.load(tmp_path / "model.zip", device="cpu")
        assert policy_digest(reloaded.policy) == expected
        assert not hasattr(reloaded, "update_writer")
    finally:
        instrumented.get_env().close()


def test_kl_brake_reduces_actual_optimizer_steps():
    torch.set_num_threads(1)
    model = make_model(ObservedMaskablePPO, target_kl=1e-12)
    rows = []
    model.update_writer = rows.append
    try:
        model.learn(64)
        assert all(row["kl_early_stopped"] for row in rows)
        assert all(0 < row["optimizer_steps"] < row["maximum_optimizer_steps"] for row in rows)
        # The upstream nominal counter still reports the full requested epochs.
        assert model._n_updates == 8
    finally:
        model.get_env().close()


def test_evaluation_rng_is_restored_even_on_error():
    import random

    model = make_model(ObservedMaskablePPO)
    model.policy.set_training_mode(True)
    try:
        random.seed(32)
        np.random.seed(32)
        torch.manual_seed(32)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        random.seed(32)
        np.random.seed(32)
        torch.manual_seed(32)
        with pytest.raises(RuntimeError), isolated_evaluation_rng(model.policy):
            random.random(), np.random.rand(), torch.rand(1)
            model.policy.set_training_mode(False)
            raise RuntimeError("failed eval")
        observed = (random.random(), np.random.rand(), torch.rand(1))
        assert observed[:2] == expected[:2] and torch.equal(observed[2], expected[2])
        assert model.policy.training
    finally:
        model.get_env().close()


@pytest.fixture
def small_config(tmp_path):
    config = TrainingConfig(
        **new_config(seed=94201, timesteps=64, output_dir=str(tmp_path / "run"))
    )
    catalog, game = config.environment()
    paths = []
    for seed in (85201, 85202):
        path = tmp_path / f"pool{seed}.json"
        build_scripted_league("midgame_stats", 1, seed, catalog=catalog, config=game).save(path)
        paths.append(str(path))
    return replace(
        config,
        environments=1,
        rollout_steps=32,
        batch_size=16,
        evaluation_interval=32,
        validation_episodes=2,
        opponent_leagues=(paths[0],),
        validation_leagues={"a": paths[1]},
    )


def test_runner_post_update_bindings_budget_and_no_overwrite(small_config):
    output = small_config.output_dir
    result = train_guarded(small_config, GuardrailConfig(n_epochs=2, max_seconds=60))
    assert result["actual_timesteps"] == 64 and result["stop_reason"] == "budget_complete"
    assert [e["checkpoint"]["timesteps"] for e in result["evaluations"]] == [0, 32, 64]
    from pathlib import Path

    root = Path(output)
    manifest = read(root / "manifest.json")
    assert manifest["training"]["action_cost"] == 0.005
    assert manifest["guardrails"]["target_kl"] == 0.01
    for record in result["evaluations"]:
        binding = record["checkpoint"]
        assert file_digest(binding["path"]) == binding["sha256"]
        loaded = MaskablePPO.load(binding["path"], device="cpu")
        assert policy_digest(loaded.policy) == binding["policy_sha256"]
    assert result["evaluations"][-1]["checkpoint"]["policy_sha256"] == policy_digest(
        MaskablePPO.load(result["last_model"], device="cpu").policy
    )
    digest = file_digest(result["last_model"])
    with pytest.raises(FileExistsError):
        train_guarded(small_config, GuardrailConfig())
    assert file_digest(result["last_model"]) == digest


def test_runner_stops_and_keeps_earlier_eligible_not_last(small_config):
    responses = iter([suite(), suite(forced=10, successes=10), suite(forced=10, successes=10)])
    config = replace(small_config, timesteps=128)
    result = train_guarded(
        config,
        GuardrailConfig(n_epochs=2),
        evaluator=lambda *_args, **_kw: deepcopy(next(responses)),
    )
    assert result["actual_timesteps"] == 64
    assert result["stop_reason"] == "validation_regression"
    assert result["selected"]["checkpoint"]["timesteps"] == 0
    assert result["selected"]["checkpoint"]["path"] != result["last_model"]


def test_weights_only_initialization_and_input_preservation(small_config, tmp_path):
    source = imitation.fresh_model(small_config)
    try:
        source.save(tmp_path / "initial.zip")
        expected = policy_digest(source.policy)
    finally:
        source.get_env().close()
    path = tmp_path / "initial.zip"
    before = file_digest(path)
    config = replace(
        small_config, initialize_from=str(path), expected_initial_policy_sha256=expected
    )
    result = train_guarded(config, GuardrailConfig(n_epochs=1))
    assert result["evaluations"][0]["checkpoint"]["policy_sha256"] == expected
    assert file_digest(path) == before
