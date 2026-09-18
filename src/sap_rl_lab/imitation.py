"""Script demonstration collection and actor-only masked behavior cloning.

No environment/reward modifications, expert win filtering, or value pretraining.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict

import numpy as np

from .baselines import scripted_policy
from .catalog import catalog_digest
from .env import SapAutoBattlerEnv
from .evaluation import file_digest
from .expanded_confirmation import save_new
from .opponents import OpponentMixture, SnapshotLeague
from .training import policy_digest


def make_env(config):
    catalog, game = config.environment()
    return SapAutoBattlerEnv(
        catalog=catalog,
        config=game,
        allow_development=config.allow_development,
        opponent_provider=OpponentMixture(
            [(1.0, SnapshotLeague.load(p)) for p in config.opponent_leagues]
        )
        if config.opponent_leagues
        else None,
        action_cost=config.action_cost,
        swap_cost=config.swap_cost,
        observe_episode_actions=config.observe_episode_actions,
    )


def fresh_model(config):
    """Match the existing trainer's architecture and initialization exactly."""
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .stable_masking import StableMaskableMultiInputPolicy

    config.validate()
    torch.set_num_threads(config.torch_threads)
    env = DummyVecEnv([lambda: make_env(config)])
    model = MaskablePPO(
        StableMaskableMultiInputPolicy,
        env,
        seed=config.seed,
        learning_rate=config.learning_rate,
        n_steps=config.rollout_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.entropy_coefficient,
        device=config.device,
        verbose=0,
    )
    catalog, game = config.environment()
    model.sap_environment_contract = {
        "schema_version": 1,
        "catalog_id": catalog.catalog_id,
        "game_config": asdict(game),
        "allow_development": True,
        "catalog_sha256": catalog_digest(catalog),
        "masking_revision": "cached-probs-clear-v1",
    }
    return model


def collect(config, teachers, episodes_per_teacher, seed, destination):
    """Store pre-action observations, legal masks and script labels; include losses."""
    if episodes_per_teacher < 1 or not teachers:
        raise ValueError("Nonempty teachers and positive episode count required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    observations, masks, actions, episode_ids, teacher_ids = [], [], [], [], []
    episodes, counts = [], Counter()
    env = make_env(config)
    try:
        for teacher_id, teacher in enumerate(teachers):
            policy = scripted_policy(teacher)
            for episode in range(episodes_per_teacher):
                episode_seed = seed + teacher_id * 100_000 + episode
                rng = random.Random(episode_seed + 1_000_003)
                obs, _ = env.reset(seed=episode_seed)
                start, forced = len(actions), 0
                while True:
                    mask = env.action_masks().copy()
                    action = env.engine.codec.encode(policy.choose(env.engine, rng))
                    if not mask[action]:
                        raise ValueError("Teacher chose illegal action")
                    observations.append({k: v.copy() for k, v in obs.items()})
                    masks.append(mask)
                    actions.append(action)
                    episode_ids.append(episode_seed)
                    teacher_ids.append(teacher_id)
                    counts[env.engine.codec.decode(action).kind.value] += 1
                    obs, _, terminated, truncated, info = env.step(action)
                    forced += int(info.get("forced_end_turn", False))
                    if terminated or truncated:
                        break
                episodes.append(
                    {
                        "seed": episode_seed,
                        "teacher": teacher,
                        "start": start,
                        "end": len(actions),
                        "wins": env.engine.state.wins,
                        "forced_turns": forced,
                        "truncated": bool(truncated),
                        "reason": info.get("reason"),
                    }
                )
    finally:
        env.close()
    data = {"obs__" + k: np.stack([o[k] for o in observations]) for k in observations[0]}
    data.update(
        masks=np.stack(masks),
        actions=np.asarray(actions, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        teacher_ids=np.asarray(teacher_ids, dtype=np.int64),
    )
    validate_data(data)
    with destination.open("xb") as handle:
        np.savez_compressed(handle, **data)
    metadata = {
        "path": str(destination),
        "sha256": file_digest(destination),
        "rows": len(actions),
        "episodes": episodes,
        "teachers": list(teachers),
        "action_counts": dict(counts),
        "selection": "All teacher actions and episodes, no success filtering",
        "rewards_used_for_labels": False,
    }
    save_new(destination.with_suffix(".json"), metadata)
    return metadata


def validate_data(data):
    n = len(data["actions"])
    masks, actions = data["masks"], data["actions"]
    if not n or masks.ndim != 2 or masks.dtype != np.bool_:
        raise ValueError("Invalid or empty demonstration masks")
    if actions.ndim != 1 or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("Invalid action labels")
    if (actions < 0).any() or (actions >= masks.shape[1]).any():
        raise ValueError("Action label out of range")
    if any(len(a) != n for a in data.values()) or not masks[np.arange(n), actions].all():
        raise ValueError("Illegal labels or misaligned arrays")
    if not any(k.startswith("obs__") for k in data):
        raise ValueError("Missing observations")
    if any(not np.isfinite(v).all() for k, v in data.items() if k.startswith("obs__")):
        raise ValueError("Nonfinite observations")


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    validate_data(data)
    return data


def validate_split(train, validation):
    if set(train["episode_ids"]) & set(validation["episode_ids"]):
        raise ValueError("Demonstration episode leakage between train and validation")
    if train.keys() != validation.keys():
        raise ValueError("Different demonstration schemas")


def actor_parameters(policy):
    """This policy has parameter-free feature flattening and disjoint actor/value MLPs."""
    selected = []
    for name, parameter in policy.named_parameters():
        if name.startswith(("mlp_extractor.policy_net.", "action_net.")):
            selected.append(parameter)
        elif not name.startswith(("mlp_extractor.value_net.", "value_net.")):
            raise ValueError(f"Unreviewed shared/trainable feature parameters: {name}")
    if not selected:
        raise ValueError("No actor parameters")
    return selected


def batch_loss(policy, data, indices):
    import torch

    obs = {
        k[5:]: torch.as_tensor(v[indices], device=policy.device)
        for k, v in data.items()
        if k.startswith("obs__")
    }
    labels = torch.as_tensor(data["actions"][indices], device=policy.device)
    distribution = policy.get_distribution(obs, action_masks=data["masks"][indices])
    loss = -distribution.log_prob(labels).mean()
    correct = (distribution.get_actions(deterministic=True) == labels).sum().item()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite BC loss")
    return loss, correct


def score(policy, data, batch_size=1024):
    import torch

    policy.set_training_mode(False)
    n, total_loss, correct = len(data["actions"]), 0.0, 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            indices = np.arange(start, min(start + batch_size, n))
            loss, hits = batch_loss(policy, data, indices)
            total_loss += loss.item() * len(indices)
            correct += hits
    return {"cross_entropy": total_loss / n, "action_accuracy": correct / n, "rows": n}


def fit(model, train_data, validation_data, folder, *, epochs=20, batch_size=512, lr=3e-4, seed=0):
    """Select by held-out imitation loss only. Critic and PPO optimizer remain untouched."""
    import torch

    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Positive BC settings required")
    validate_data(train_data)
    validate_data(validation_data)
    validate_split(train_data, validation_data)
    folder.mkdir(parents=True, exist_ok=False)
    actor = actor_parameters(model.policy)
    critic_before = {
        k: v.detach().clone()
        for k, v in model.policy.state_dict().items()
        if k.startswith(("mlp_extractor.value_net.", "value_net."))
    }
    optimizer = torch.optim.Adam(actor, lr=lr, eps=1e-5)
    rng = np.random.default_rng(seed)
    history, best, selected = [], float("inf"), None
    for epoch in range(epochs + 1):
        if epoch:
            model.policy.set_training_mode(True)
            order = rng.permutation(len(train_data["actions"]))
            for start in range(0, len(order), batch_size):
                loss, _ = batch_loss(model.policy, train_data, order[start : start + batch_size])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor, 1.0, error_if_nonfinite=True)
                optimizer.step()
        row = {
            "epoch": epoch,
            "train": score(model.policy, train_data),
            "validation": score(model.policy, validation_data),
        }
        path = folder / f"epoch{epoch:03d}.zip"
        model.save(str(path))
        row.update(
            path=str(path), sha256=file_digest(path), policy_sha256=policy_digest(model.policy)
        )
        if epoch and row["validation"]["cross_entropy"] < best:
            best, selected = row["validation"]["cross_entropy"], row
        history.append(row)
        save_new(folder / f"epoch{epoch:03d}.json", row)
        print(
            f"BC epoch={epoch} val_loss={row['validation']['cross_entropy']:.4f} "
            f"accuracy={row['validation']['action_accuracy']:.3f}",
            flush=True,
        )
    if any(not torch.equal(v, model.policy.state_dict()[k]) for k, v in critic_before.items()):
        raise ValueError("BC unexpectedly changed the critic")
    if model.policy.optimizer.state:
        raise ValueError("BC unexpectedly populated PPO optimizer state")
    summary = {
        "history": history,
        "selected": selected,
        "critic_unchanged": True,
        "ppo_optimizer_untouched": True,
        "epochs": epochs,
        "training_example_visits": epochs * len(train_data["actions"]),
        "gradient_updates": epochs * ((len(train_data["actions"]) + batch_size - 1) // batch_size),
    }
    save_new(folder / "fit.json", summary)
    return summary
