"""Real, tiny BC -> KL-PPO -> exact reload -> opponent snapshot integration check."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from sb3_contrib import MaskablePPO

from sap_rl_lab.fullpack import imitation
from sap_rl_lab.fullpack.catalog import catalog_digest
from sap_rl_lab.fullpack.evaluation import evaluate_suite, file_digest
from sap_rl_lab.fullpack.learned_pool import collect as collect_learned
from sap_rl_lab.fullpack.opponents import SnapshotLeague, build_scripted_league
from sap_rl_lab.fullpack.ppo_guardrails import GuardrailConfig, train_guarded, write_new
from sap_rl_lab.fullpack.recipe import FAMILIES, LearnedFirstGuard, configuration
from sap_rl_lab.fullpack.training import policy_digest
from sap_rl_lab.round3 import source_archive


def smoke(output, tier=6):
    output.mkdir(parents=True, exist_ok=False)
    config = replace(
        configuration(tier=tier, seed=83101), environments=2, rollout_steps=8, batch_size=8
    )
    catalog, game = config.environment()
    pool_path = output / "train-pool.json"
    build_scripted_league(FAMILIES[0], 2, 83000000, catalog=catalog, config=game).save(pool_path)
    config = replace(config, opponent_leagues=(str(pool_path),))
    train_path, val_path = output / "train.npz", output / "validation.npz"
    imitation.collect(config, FAMILIES, 2, 83100000, train_path)
    imitation.collect(config, FAMILIES, 1, 84100000, val_path)
    model = imitation.fresh_model(config)
    try:
        parameters = sum(p.numel() for p in model.policy.parameters())
        shapes = {k: list(v.shape) for k, v in model.observation_space.spaces.items()}
        bc = imitation.fit(
            model,
            imitation.load_data(train_path),
            imitation.load_data(val_path),
            output / "bc",
            epochs=2,
            seed=config.seed,
        )
    finally:
        model.get_env().close()
    families = {}
    for i in range(2):
        target = output / f"validation-pool-{i}.json"
        build_scripted_league(
            FAMILIES[i], 2, 83300000 + i * 10000, catalog=catalog, config=game
        ).save(target)
        families[f"validation_{i}"] = str(target)
    config = replace(
        config,
        timesteps=32,
        output_dir=str(output / "ppo"),
        initialize_from=bc["selected"]["path"],
        expected_initial_policy_sha256=bc["selected"]["policy_sha256"],
        validation_leagues=families,
        validation_episodes=2,
        validation_seed=83400000,
        evaluation_interval=16,
    )
    result = train_guarded(
        config, GuardrailConfig(stop_on_regression=False), guard_factory=LearnedFirstGuard
    )
    assert result["actual_timesteps"] == 32 and result["stop_reason"] == "budget_complete"
    final = MaskablePPO.load(result["last_model"], device="cpu")
    binding = result["evaluations"][-1]["checkpoint"]
    assert final.target_kl == 0.01 and policy_digest(final.policy) == binding["policy_sha256"]
    assert file_digest(binding["path"]) == binding["sha256"]
    expected = json.loads(Path(binding["path"]).with_suffix(".json").read_text())["evaluation"]
    actual = evaluate_suite(final, families, episodes=2, seed=83400000)
    for family in families:
        assert (
            actual["families"][family]["episode_results"]
            == expected["families"][family]["episode_results"]
        )
    pool, rows = collect_learned(
        final, episodes=1, seed=83500000, opponent_provider=SnapshotLeague.load(pool_path)
    )
    assert len(pool) == rows[0]["battles"] and not rows[0]["truncated"]
    pool.save(output / "learned.json")
    summary = dict(
        ppo_decisions=32,
        exact_reload=True,
        reload_episodes=4,
        parameters=parameters,
        shapes=shapes,
        actions=int(final.action_space.n),
        target_kl=final.target_kl,
        learned_snapshots=len(pool),
        catalog_id=catalog.catalog_id,
        catalog_sha256=catalog_digest(catalog),
        source_files_sha256=source_archive(output),
    )
    write_new(output / "summary.json", summary)
    print({k: v for k, v in summary.items() if k != "source_files_sha256"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", type=int, choices=(5, 6), default=6)
    args = parser.parse_args()
    smoke(args.output.resolve(), args.tier)
