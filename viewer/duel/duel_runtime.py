"""Human/model duel adapter. Uses frozen rules, observation and settlement.

Runs unchanged in CPython tests and a Pyodide Web Worker. No Torch, network,
training, pickles or arbitrary user-code execution in the browser.
"""

import json
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .catalog import catalog_digest, catalog_from_dict
from .domain import BattleOutcome, GameConfig
from .engine import AutoBattler, BattleResult, Transition
from .generated import Observation, finish_battle
from .late_events import late_battle_runtime


class NumpyPolicy(Observation):
    def __init__(self, root):
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.catalog_raw = json.loads((root / "catalog.json").read_text())
        self.catalog = catalog_from_dict(self.catalog_raw)
        assert catalog_digest(self.catalog) == self.manifest["catalog_sha256"]
        self.config = GameConfig(**self.manifest["config"])
        self.weights = dict(np.load(root / "model.npz", allow_pickle=False))
        self.observation_space = {
            k: SimpleNamespace(shape=tuple(v))
            for k, v in self.manifest["observation_shapes"].items()
        }
        self._pet_index = {name: i for i, name in enumerate(self.catalog.pets)}
        self._food_index = {name: i for i, name in enumerate(self.catalog.foods)}
        self.observe_episode_actions = False

    def observe(self, engine):
        self.engine = engine
        return self._observation()

    def forward(self, observation, mask):
        x = np.concatenate(
            [observation[k].ravel() for k in self.manifest["observation_order"]]
        ).astype(np.float32)

        def linear(y, prefix):
            # einsum avoids spurious Apple Accelerate floating-point flags and
            # is also available in Pyodide's NumPy, without BLAS threading.
            return (
                np.einsum("ij,j->i", self.weights[prefix + ".weight"], y)
                + self.weights[prefix + ".bias"]
            )

        def hidden(prefix):
            return np.tanh(linear(np.tanh(linear(x, prefix + ".0")), prefix + ".2"))

        logits = linear(hidden("mlp_extractor.policy_net"), "action_net")
        value = float(linear(hidden("mlp_extractor.value_net"), "value_net")[0])
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("No legal actions in finished state")
        logits = np.where(mask, logits, np.float32(-1e8))
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return int(np.argmax(probabilities)), probabilities, value

    def predict(self, engine):
        return self.forward(self.observe(engine), engine.action_mask())


class DuelSide(AutoBattler):
    ready = False

    def legal_actions(self):
        return () if self.ready else super().legal_actions()

    def _end_turn(self):
        self._shop_runtime().phase("end_turn")
        self.ready = True
        return Transition(0.0, False, False, {"waiting_for_opponent": True})

    def finish(self, result):
        transition = finish_battle(self, result)
        self.ready = False
        return transition


class ObservedTrace(list):
    """Read-only snapshots at the existing resolver's trace boundaries."""

    def __init__(self, runtime, frames):
        super().__init__()
        self.runtime, self.frames = runtime, frames

    def append(self, message):
        super().append(message)
        self.frames.append(
            {
                "message": message,
                "teams": [[asdict(p) for p in team] for team in self.runtime.teams],
            }
        )


class Duel:
    def __init__(self, policy, seed):
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be a 32-bit unsigned integer")
        self.policy, self.seed = policy, seed
        self.human = DuelSide(policy.catalog, policy.config)
        self.ai = DuelSide(policy.catalog, policy.config)
        self.human.reset(seed)
        self.ai.reset(seed ^ 0x9E3779B9)
        self.battle_rng = random.Random(seed ^ 0x85EBCA6B)
        self.done = False
        self.result = None
        self.battle = None
        self.phase = "shop"
        self.revision = 0
        self.ai_decisions = []
        self.history = []
        self.human_forced = False
        self.ai_forced = False

    def act(self, action_id, revision):
        if self.done or self.phase != "shop" or revision != self.revision:
            raise ValueError("This action belongs to an old or finished turn")
        if type(action_id) is not int or action_id not in self.human.legal_action_ids():
            raise ValueError("Illegal action")
        # A failed adapter transition must not leave half a turn applied.
        before = deepcopy({k: v for k, v in self.__dict__.items() if k != "policy"})
        try:
            label = self.human.codec.describe(action_id)
            transition = self.human.step_id(action_id)
            self.human_forced = bool(transition.info.get("forced_end_turn"))
            if self.human.ready:
                self._battle()
        except Exception:
            self.__dict__.update(before)
            raise
        self.history.append({"turn": before["human"].state.turn, "action": label, "id": action_id})
        self.revision += 1
        return self.view()

    def _battle(self):
        self.ai_decisions = []
        for _ in range(self.ai.config.max_actions_per_turn + 1):
            action, probabilities, value = self.policy.predict(self.ai)
            legal = self.ai.legal_action_ids()
            top = sorted(legal, key=lambda i: -float(probabilities[i]))[:5]
            self.ai_decisions.append(
                {
                    "action": self.ai.codec.describe(action),
                    "label": self.action_label(action),
                    "value": value,
                    "top": [
                        {
                            "action": self.ai.codec.describe(i),
                            "label": self.action_label(i),
                            "probability": float(probabilities[i]),
                        }
                        for i in top
                    ],
                }
            )
            transition = self.ai.step_id(action)
            self.ai_forced = bool(transition.info.get("forced_end_turn"))
            if self.ai.ready:
                break
        if not self.ai.ready:
            raise RuntimeError("Model did not finish within the action budget")
        runtime = late_battle_runtime(
            self.human.state.team, self.ai.state.team, self.policy.catalog, self.battle_rng
        )
        frames = [
            {
                "message": "双方回合结束效果已结算 · 开始战斗",
                "teams": [[asdict(p) for p in team] for team in runtime.teams],
            }
        ]
        runtime.trace = ObservedTrace(runtime, frames)
        outcome, attacks = runtime.battle()
        self.battle = {
            "turn": self.human.state.turn,
            "outcome": outcome.name.lower(),
            "frames": frames,
            "trace": list(runtime.trace),
            "attacks": attacks,
            "attack_limit_reached": runtime.attack_limit_reached,
            "human_forced": self.human_forced,
            "ai_forced": self.ai_forced,
            "human_team": [asdict(p) for p in self.human.state.team],
            "ai_team": [asdict(p) for p in self.ai.state.team],
        }
        human_result = BattleResult(
            outcome, list(runtime.trace), attacks, runtime.attack_limit_reached
        )
        ai_result = BattleResult(
            BattleOutcome(-int(outcome)), [], attacks, runtime.attack_limit_reached
        )
        human_transition, ai_transition = self.human.finish(human_result), self.ai.finish(ai_result)
        self.done = any(
            [
                human_transition.terminated,
                human_transition.truncated,
                ai_transition.terminated,
                ai_transition.truncated,
            ]
        )
        if self.done:
            h, a = self.human.state, self.ai.state
            self.result = (
                "human"
                if a.lives <= 0 or h.wins >= 10
                else "ai"
                if h.lives <= 0 or a.wins >= 10
                else "turn_limit"
            )
        self.phase = "battle"

    def action_label(self, action_id):
        action = self.ai.codec.decode(action_id)
        source, target, kind = action.source, action.target, action.kind.value

        def pet_name(index):
            return self.policy.catalog.pets[self.ai.state.team[index].spec_id].name

        if kind == "end_turn":
            return "结束回合"
        if kind == "roll":
            return "刷新商店"
        if kind in {"buy_pet", "buy_food", "merge", "toggle_freeze"}:
            item = self.ai.state.shop[source]
            specs = self.policy.catalog.pets if item.kind == "pet" else self.policy.catalog.foods
            label = specs[item.item_id].name
            if kind == "buy_pet":
                return f"购买 {label}"
            if kind == "buy_food":
                return f"{label} → {target} 号 {pet_name(target)}"
            if kind == "merge":
                return f"买入合成 {label} → {target} 号"
            return f"{'解冻' if item.frozen else '冻结'} {label}"
        if kind == "sell":
            return f"卖出 {source} 号 {pet_name(source)}"
        if kind == "swap_adjacent":
            return f"交换 {source} 号 {pet_name(source)} ↔ {target} 号 {pet_name(target)}"
        return f"合成 {source} 号 {pet_name(source)} → {target} 号"

    def next_round(self, revision):
        if self.done or self.phase != "battle" or revision != self.revision:
            raise ValueError("No next shop available")
        self.phase = "shop"
        self.human_forced = self.ai_forced = False
        self.revision += 1
        return self.view()

    def view(self):
        human = self.human.state.to_dict()
        # Never disclose the AI's current shop. Show only its last revealed team.
        return {
            "seed": self.seed,
            "revision": self.revision,
            "phase": self.phase,
            "done": self.done,
            "result": self.result,
            "human": human,
            "ai": {
                "lives": self.ai.state.lives,
                "wins": self.ai.state.wins,
                "team": self.battle["ai_team"] if self.battle else [],
            },
            "legal": [
                {"id": i, **asdict(self.human.codec.decode(i))}
                for i in self.human.legal_action_ids()
            ]
            if self.phase == "shop" and not self.done
            else [],
            "battle": self.battle,
            "ai_decisions": self.ai_decisions,
        }


_policy = None
_duel = None


def initialize(root):
    global _policy
    _policy = NumpyPolicy(root)
    # Check real browser arithmetic against golden Torch outputs, fail closed.
    with np.load(Path(root) / "parity.npz", allow_pickle=False) as fixture:
        count = len(fixture["action"])
        for i in range(count):
            obs = {k: fixture[k][i] for k in _policy.manifest["observation_order"]}
            action, probs, value = _policy.forward(obs, fixture["mask"][i])
            if (
                action != int(fixture["action"][i])
                or np.max(abs(probs - fixture["probabilities"][i])) > 2e-5
                or abs(value - float(fixture["value"][i])) > 2e-5
            ):
                raise ValueError("Browser inference parity check failed")
    return json.dumps(
        {"manifest": _policy.manifest, "catalog": _policy.catalog_raw, "parity_cases": count}
    )


def dispatch(message):
    global _duel
    request = json.loads(message)
    if request["type"] == "start":
        _duel = Duel(_policy, request["seed"])
        result = _duel.view()
    elif request["type"] == "action" and _duel:
        result = _duel.act(request["action"], request["revision"])
    elif request["type"] == "next" and _duel:
        result = _duel.next_round(request["revision"])
    else:
        raise ValueError("Unknown game request")
    return json.dumps(result)
