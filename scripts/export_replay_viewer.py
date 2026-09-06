"""Export observed Round 3 trajectories using its hash-verified archived engine.

No training and no game-rule edits. A line observer captures battle states without
consuming randomness. Selected-model values are diagnostic reconstructions, not
the historical rollout-buffer values used by the optimizer.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, separators=(",", ":"))


class BattleObserver:
    """Read-only observation of trace appends inside the archived resolver."""

    def __init__(self, engine_file):
        self.engine_file = str(Path(engine_file).resolve())
        self.file_matches = {}
        self.frames = []
        self.count = 0

    def __call__(self, frame, event, arg):
        filename = frame.f_code.co_filename
        if filename not in self.file_matches:
            self.file_matches[filename] = str(Path(filename).resolve()) == self.engine_file
        if not self.file_matches[filename]:
            return None
        local = frame.f_locals
        if event in {"line", "return"} and "teams" in local and "trace" in local:
            trace, teams = local["trace"], local["teams"]
            if not self.frames or len(trace) > self.count:
                self.frames.append(
                    {
                        "event": trace[-1] if trace else "Battle begins",
                        "teams": [[copy.deepcopy(vars(p)) for p in team] for team in teams],
                    }
                )
                self.count = len(trace)
        return self


def select_cases(root, selection):
    cases = []
    order = [
        "mixed-seed307",
        "mixed-seed211",
        "mixed-seed101",
        "single-seed307",
        "single-seed211",
        "single-seed101",
    ]
    order = [n for n in order if n in selection]
    for number, name in enumerate(order):
        families = ["stats", ("greedy" if number % 2 else "summon")]
        first_failure = None
        for family in families:
            report = json.loads((root / f"test-{name}-{family}.json").read_text())
            failures = sorted(
                (r for r in report["episode_results"] if r["truncated"]), key=lambda r: r["seed"]
            )
            if not failures:
                raise ValueError(f"No cutoff episode in {name}/{family}")
            row = failures[0]
            first_failure = first_failure or row
            cases.append((name, family, "cutoff", row))
        report = json.loads((root / f"test-{name}-stats.json").read_text())
        row = min(
            (r for r in report["episode_results"] if r["success"]),
            key=lambda r: (abs(r["final_turn"] - first_failure["final_turn"]), r["seed"]),
        )
        cases.append((name, "stats", "success", row))
    return cases


def select_confirmation_cases(root, selection):
    """Up to two smallest-seed cutoffs and one success per model, plus a paired example."""
    cases, reports = [], {}
    for name in selection:
        reports[name] = json.loads((root / f"test-{name}.json").read_text())["families"]
        failures = []
        # One case per family before taking a second from a family.
        for family in ("stats", "summon", "greedy"):
            rows = reports[name][family]["episode_results"]
            cutoffs = sorted((r for r in rows if r["truncated"]), key=lambda r: r["seed"])
            failures.extend((rank, family, row) for rank, row in enumerate(cutoffs))
        failures.sort(key=lambda x: (x[0], ("stats", "summon", "greedy").index(x[1])))
        for _, family, row in failures[:2]:
            cases.append((name, family, "cutoff", row))
        target_turn = failures[0][2]["final_turn"] if failures else 10
        wins = [r for r in reports[name]["stats"]["episode_results"] if r["success"]]
        if wins:
            row = min(wins, key=lambda r: (abs(r["final_turn"] - target_turn), r["seed"]))
            cases.append((name, "stats", "success", row))
    release = json.loads((root / "release" / "manifest.json").read_text())
    improved = release["model"]
    control = f"unshaped-seed{release['seed']}"
    default_ids = None
    for family in ("stats", "summon", "greedy"):
        left = reports[control][family]["episode_results"]
        right = reports[improved][family]["episode_results"]
        for a, b in zip(left, right):
            if a["seed"] != b["seed"]:
                raise ValueError("Unpaired test rows")
            if a["truncated"] and b["success"]:
                pair = [(control, family, "cutoff", a), (improved, family, "success", b)]
                for case in pair:
                    if not any(
                        (n, f, r["seed"]) == (case[0], case[1], case[3]["seed"])
                        for n, f, _, r in cases
                    ):
                        cases.append(case)
                default_ids = [f"{n}-{f}-{r['seed']}" for n, f, _, r in pair]
                break
        if default_ids:
            break
    return cases, default_ids


def artifact_path(root, value):
    """Resolve original absolute metadata or a public repository-relative export."""
    path = Path(value)
    return path if path.is_absolute() else root.parent.parent / path


def worker(root, output):
    import torch
    from sb3_contrib import MaskablePPO

    import sap_rl_lab.engine as engine_module
    from sap_rl_lab.env import SapAutoBattlerEnv
    from sap_rl_lab.opponents import SnapshotLeague

    torch.set_num_threads(1)
    protocol = json.loads((root / "protocol.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    confirmation = protocol.get("round") == 5
    cases, default_ids = (
        select_confirmation_cases(root, selection)
        if confirmation
        else (select_cases(root, selection), None)
    )
    entries = []
    model = None
    current = None
    for name, family, category, row in cases:
        if name != current:
            checkpoint = artifact_path(root, selection[name]["path"])
            if digest(checkpoint) != selection[name]["sha256"]:
                raise ValueError("Checkpoint changed")
            model = MaskablePPO.load(checkpoint, device="cpu")
            model.policy.set_training_mode(False)
            current = name
            training_config = json.loads((root / name / "run_manifest.json").read_text())["config"]
            swap_cost = training_config.get("swap_cost", 0.0)
        league_path = artifact_path(root, protocol["paths"]["test"][family])
        relative = str(Path(league_path).relative_to(root))
        if digest(league_path) != protocol["data_sha256"][relative]:
            raise ValueError("League changed")
        provider = SnapshotLeague.load(league_path)
        env_kwargs = {"observe_episode_actions": True} if confirmation else {}
        env = SapAutoBattlerEnv(opponent_provider=provider, **env_kwargs)
        observation, _ = env.reset(seed=row["seed"])
        steps = []
        rewards = 0.0
        while True:
            state = copy.deepcopy(env.engine.state.to_dict())
            tensor, _ = model.policy.obs_to_tensor(observation)
            with torch.no_grad():
                dist = model.policy.get_distribution(tensor, action_masks=env.action_masks())
                probabilities = dist.distribution.probs.cpu().numpy()[0]
                value = float(model.policy.predict_values(tensor).item())
            chosen, _ = model.predict(
                observation, action_masks=env.action_masks(), deterministic=True
            )
            action_id = int(chosen)
            legal = [
                {
                    "id": a,
                    "label": env.engine.codec.describe(a),
                    "probability": float(probabilities[a]),
                }
                for a in env.engine.legal_action_ids()
            ]
            legal.sort(key=lambda a: (-a["probability"], a["id"]))
            observer = BattleObserver(engine_module.__file__)
            previous_trace = sys.gettrace()
            if env.engine.codec.describe(action_id) == "end_turn":
                sys.settrace(observer)
            try:
                next_observation, reward, terminated, truncated, info = env.step(action_id)
            finally:
                sys.settrace(previous_trace)
            if env.engine.codec.describe(action_id) == "end_turn" and not observer.frames:
                raise AssertionError("Battle observation failed; refusing an incomplete replay")
            tensor, _ = model.policy.obs_to_tensor(next_observation)
            with torch.no_grad():
                next_value = float(model.policy.predict_values(tensor).item())
            bootstrap = model.gamma * next_value if truncated and not terminated else 0.0
            objective_reward = reward - (
                swap_cost
                if env.engine.codec.describe(action_id).startswith("swap_adjacent:")
                else 0.0
            )
            rewards += reward
            steps.append(
                {
                    "index": len(steps),
                    "state": state,
                    "after": copy.deepcopy(env.engine.state.to_dict()),
                    "action": env.engine.codec.describe(action_id),
                    "action_id": action_id,
                    "legal_actions": legal,
                    "value": value,
                    "next_value": next_value,
                    "reward": reward,
                    "objective_reward": objective_reward,
                    "total_reward": rewards,
                    "terminated": terminated,
                    "truncated": truncated,
                    "reason": info.get("reason"),
                    "outcome": info.get("battle_outcome"),
                    "bootstrap": bootstrap,
                    "rollout_reward": objective_reward + bootstrap,
                    "td_target": objective_reward
                    + (0.0 if terminated else model.gamma * next_value),
                    "battle": observer.frames,
                }
            )
            observation = next_observation
            if terminated or truncated:
                break
        actual = {
            "return": rewards,
            "actions": len(steps),
            "wins": env.engine.state.wins,
            "truncated": truncated,
            "success": env.engine.state.wins >= 10,
        }
        for key, value in actual.items():
            if value != row[key]:
                raise AssertionError((name, row["seed"], key, value, row[key]))
        remaining = objective_remaining = 0.0
        for step in reversed(steps):
            remaining += step["reward"]
            objective_remaining += step["objective_reward"]
            step["remaining_return"] = remaining
            step["objective_remaining_return"] = objective_remaining
        identifier = f"{name}-{family}-{row['seed']}"
        entry = {
            "id": identifier,
            "policy": name,
            "family": family,
            "category": category,
            "seed": row["seed"],
            "wins": actual["wins"],
            "actions": len(steps),
            "turns": row["final_turn"],
            "file": f"replays/{identifier}.json",
        }
        payload = {
            **entry,
            "schema_version": 1,
            "verified": True,
            "catalog_id": env.engine.catalog.catalog_id,
            "model_sha256": selection[name]["sha256"],
            "league_sha256": digest(league_path),
            "gamma": model.gamma,
            "swap_cost": swap_cost,
            "objective_note": "Critic predicts training reward: raw game reward "
            f"minus {swap_cost} per swap.",
            "steps": steps,
            "final_state": env.engine.state.to_dict(),
        }
        save(output / entry["file"], payload)
        entries.append(entry)
        env.close()
        print(f"EXPORTED {identifier}: {category}, {len(steps)} actions", flush=True)
    catalog = SapAutoBattlerEnv().engine.catalog
    manifest = {
        "schema_version": 1,
        "title": "Round 3 · Frozen-rule replay collection",
        "rules": "Round 3 historical rules (includes the old Fish level-up bug)",
        "diagnostic_note": "Probabilities and values reconstructed from the selected checkpoint; "
        "not recorded historical training-buffer values. Bootstrap is what PPO would add "
        "at this checkpoint. This is evidence to inspect, not a proven cause of looping.",
        "selection": "Two smallest-seed cutoff cases per model across two families; one stats "
        "success nearest the first failure's final turn. Twelve cutoffs and six successes. "
        "Pairs are illustrative, not controlled causal comparisons.",
        "source_archive_sha256": digest(root / "source.zip"),
        "pets": {
            key: {
                "name": p.name,
                "attack": p.attack,
                "health": p.health,
                "abilities": [
                    {"trigger": a.trigger, "effect": a.effect, "params": dict(a.params)}
                    for a in p.abilities
                ],
            }
            for key, p in catalog.pets.items()
        },
        "episodes": entries,
    }
    if confirmation:
        summary = json.loads((root / "summary.json").read_text())
        manifest.update(
            title="Round 5 · Equal-budget confirmation",
            rules="Corrected Fish · rules-v2",
            diagnostic_note="Selected-checkpoint reconstructions, not original training-buffer "
            "values. Critic and realized objective return include that model's swap penalty; "
            "game rewards and reported win/cutoff rates do not.",
            selection="Up to two smallest-seed cutoffs across families and one stats success "
            "nearest the first failure's final turn per model. Default pair: smallest-seed "
            "control cutoff / delivery-model success, scanning stats, summon, greedy. Same "
            "initial episode seed and opponent pool, not identical later states. Examples "
            "are deliberately selected, not a representative sample or proof of causality.",
            default_ids=default_ids,
            benchmark=summary["arms"],
            delivery_model=summary["delivery_model"],
            checks=summary["predeclared_checks"],
        )
    save(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("output_directory")
    parser.add_argument("--archived-worker", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.run_directory).resolve(), Path(args.output_directory).resolve()
    if args.archived_worker:
        worker(root, output)
        return
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((root / "protocol.json").read_text())
    with tempfile.TemporaryDirectory(prefix="sap-round3-archive-") as temporary:
        archive_root = Path(temporary)
        with zipfile.ZipFile(root / "source.zip") as archive:
            for relative, expected in protocol["source_files_sha256"].items():
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe archived path")
                data = archive.read(relative)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("Archived source hash differs")
                destination = archive_root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        env = dict(os.environ, PYTHONPATH=str(archive_root / "src"))
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(root),
                str(output),
                "--archived-worker",
            ],
            env=env,
            check=True,
            cwd=temporary,
        )


if __name__ == "__main__":
    main()
