"""Reachable checkpoint opponent snapshots, captured at the actual battle boundary.

Generation is not evaluation of any candidate against the resulting pool.
Keep challenge pools out of candidate training/validation and freeze selections
before opening their scores. Never call the legacy eight-pet pool builder for v4.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from ..experiments import save_json
from ..round3 import league_profile, source_archive
from .evaluation import file_digest, policy_environment_options
from .opponents import SnapshotLeague


def collect(model, *, episodes, seed, opponent_provider, snapshot_metadata=None):
    from .env import SapAutoBattlerEnv

    if episodes < 1:
        raise ValueError("episodes must be positive")
    options = policy_environment_options(model)
    if not options or options["catalog"].rules_version not in {4, 5, 6, 7, 8}:
        raise ValueError("Opponent generation requires a supported saved environment contract")
    league = SnapshotLeague(catalog_id=options["catalog"].catalog_id)
    episode_seed = seed

    def capture(turn, rng, catalog, config):
        label = f"learned-seed-{episode_seed}"
        league.add(turn, env.engine.state.team, label=label)
        if snapshot_metadata is not None:
            state = env.engine.state
            snapshot_metadata.append(
                {
                    "label": label,
                    "episode_seed": episode_seed,
                    "turn": turn,
                    "wins_before_battle": state.wins,
                    "lives_before_battle": state.lives,
                    "gold": state.gold,
                    "actions_this_turn": state.actions_this_turn,
                }
            )
        return opponent_provider(turn, rng, catalog, config)

    env = SapAutoBattlerEnv(opponent_provider=capture, **options)
    rows = []
    try:
        for episode_seed in range(seed, seed + episodes):
            observation, _ = env.reset(seed=episode_seed)
            actions, forced, battles = 0, 0, 0
            action_counts = Counter()
            fingerprint = hashlib.sha256()
            before = len(league)
            while True:
                action, _ = model.predict(
                    observation, action_masks=env.action_masks(), deterministic=True
                )
                action = int(action)
                if snapshot_metadata is not None:
                    action_counts[env.engine.codec.decode(action).kind.value] += 1
                fingerprint.update(f"{action},".encode())
                observation, _, terminated, truncated, info = env.step(action)
                actions += 1
                forced += int(bool(info.get("forced_end_turn")))
                battles += int("battle_outcome" in info)
                if terminated or truncated:
                    break
            if len(league) - before != battles:
                raise AssertionError("Snapshots must match every actual battle exactly once")
            rows.append(
                {
                    "seed": episode_seed,
                    "actions": actions,
                    "battles": battles,
                    "forced_battles": forced,
                    "truncated": truncated,
                    "wins": env.engine.state.wins,
                    "action_sha256": fingerprint.hexdigest(),
                }
            )
            if snapshot_metadata is not None:
                rows[-1]["action_counts"] = dict(action_counts)
    finally:
        env.close()
    return league, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--against", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    model_path, against = Path(args.model).resolve(), Path(args.against).resolve()
    model_hash, against_hash = file_digest(str(model_path)), file_digest(str(against))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = source_archive(output)
    model = MaskablePPO.load(model_path, device="cpu")
    league, rows = collect(
        model,
        episodes=args.episodes,
        seed=args.seed,
        opponent_provider=SnapshotLeague.load(against),
    )
    path = output / "pool.json"
    league.save(path)
    assert file_digest(str(model_path)) == model_hash
    assert file_digest(str(against)) == against_hash
    save_json(
        output / "manifest.json",
        {
            "model": str(model_path),
            "model_sha256": model_hash,
            "against": str(against),
            "against_sha256": against_hash,
            "source_files_sha256": sources,
            "pool_sha256": file_digest(str(path)),
            "environment_contract": model.sap_environment_contract,
            "episode_results": rows,
            "profile": league_profile(path),
            "candidate_evaluation_performed": False,
            "capture": "Exactly once per actual battle, after end-turn abilities, "
            "including forced endings",
        },
    )
    print(f"Saved {len(league)} reachable learned-team snapshots; no candidate challenge evaluated")


if __name__ == "__main__":
    main()
