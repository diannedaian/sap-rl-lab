"""Gymnasium adapter kept deliberately thin around the tested game engine."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
    raise ImportError(
        'The RL adapter needs optional packages. Install with: pip install -e ".[rl]"'
    ) from exc

from .actions import ActionKind
from .catalog import Catalog, load_catalog
from .domain import GameConfig
from .engine import AutoBattler, InvalidAction, OpponentProvider
from .shop import ContentNotReady


class SapAutoBattlerEnv(gym.Env):
    """Single-agent Arena environment with dynamic invalid-action masks."""

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        catalog: Optional[Catalog] = None,
        config: Optional[GameConfig] = None,
        opponent_provider: Optional[OpponentProvider] = None,
        render_mode: Optional[str] = None,
        action_cost: float = 0.0,
        forfeit_on_limit: bool = False,
        swap_cost: float = 0.0,
        success_bonus_max: float = 0.0,
        success_action_cost: float = 0.0,
        observe_episode_actions: bool = False,
        allow_development: bool = False,
    ) -> None:
        super().__init__()
        selected_catalog = catalog or load_catalog()
        if selected_catalog.development_only and not allow_development:
            raise ValueError(
                "Development catalog is not training-ready. "
                "allow_development=True is only for explicit experimental diagnostics/training."
            )
        self.engine = AutoBattler(selected_catalog, config, opponent_provider)
        if forfeit_on_limit and self.engine.config.shop_action_limit_mode == "force_battle":
            raise ValueError(
                "force_battle and forfeit_on_limit are different environment contracts"
            )
        self.render_mode = render_mode
        costs = (action_cost, swap_cost, success_bonus_max, success_action_cost)
        if any(not math.isfinite(x) or x < 0 for x in costs):
            raise ValueError("reward coefficients must be finite and nonnegative")
        if success_bonus_max and (not success_action_cost or not observe_episode_actions):
            raise ValueError("success bonus requires a positive action cost and observed count")
        self.action_cost = action_cost
        self.forfeit_on_limit = forfeit_on_limit
        self.swap_cost = swap_cost
        self.success_bonus_max = success_bonus_max
        self.success_action_cost = success_action_cost
        self.observe_episode_actions = observe_episode_actions
        self.episode_actions = 0
        self.action_space = spaces.Discrete(self.engine.codec.size)

        pet_count = len(self.engine.catalog.pets)
        food_count = len(self.engine.catalog.foods)
        team_width = pet_count + 9
        shop_width = pet_count + food_count + 6
        if self.engine.config.shop_pet_stats:
            shop_width += 6
        if self.engine.modern_shop:
            shop_width += self.engine.config.max_shop_size
        if self.engine.expanded_rules:
            # Extra perk bits plus the remaining per-turn ability budget.
            team_width += 3
            shop_width += 3
        if self.engine.midgame_rules:
            # Garlic/Chili/Cake, accumulated sale bonus, copied-ability identity.
            team_width += 4 + pet_count
            shop_width += 4 + pet_count
        if self.engine.tier4_rules:
            team_width += 2  # Peanut and Bread.
            shop_width += 2
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(
                    0.0,
                    1.0,
                    shape=(8 + int(observe_episode_actions) + 2 * int(self.engine.tier4_rules),),
                    dtype=np.float32,
                ),
                "team": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.engine.config.max_team_size, team_width),
                    dtype=np.float32,
                ),
                "shop": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.engine.config.max_shop_size, shop_width),
                    dtype=np.float32,
                ),
            }
        )
        self._pet_index = {pet_id: index for index, pet_id in enumerate(self.engine.catalog.pets)}
        self._food_index = {
            food_id: index for index, food_id in enumerate(self.engine.catalog.foods)
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        # VecEnv autoresets call reset(seed=None). Passing None into the pure
        # engine would reseed Python's RNG from system entropy on EVERY episode,
        # defeating the training seed after the first episode. Draw subsequent
        # episode seeds from this environment's persistent seeded Gym RNG.
        episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**63 - 1))
        self.engine.reset(seed=episode_seed)
        self.episode_actions = 0
        return self._observation(), {
            "seed": episode_seed,
            "catalog_id": self.engine.catalog.catalog_id,
        }

    def step(self, action: int) -> tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.episode_actions += 1
        try:
            transition = self.engine.step_id(int(action))
        except ContentNotReady:
            self.episode_actions -= 1
            raise
        except InvalidAction as exc:
            # The pure engine fails before mutation. At the Gym boundary an
            # unmasked/random policy gets a defined transition rather than a
            # corrupted state or an exception hidden by a retry loop.
            self.engine.state.truncated = True
            return (
                self._observation(),
                -1.0,
                False,
                True,
                {"reason": "invalid_action", "error": str(exc)},
            )
        info = dict(transition.info)
        info.update(
            {
                "turn": self.engine.state.turn,
                "wins": self.engine.state.wins,
                "lives": self.engine.state.lives,
            }
        )
        # Optional training objective. Evaluation defaults preserve the original
        # game rewards, action mask, and cutoffs for an unchanged benchmark.
        reward = transition.reward - self.action_cost
        terminated, truncated = transition.terminated, transition.truncated
        info["game_reward"] = transition.reward
        info["episode_actions"] = self.episode_actions
        swap_penalty = (
            self.swap_cost if self.engine.codec.decode(int(action)).kind is ActionKind.SWAP else 0.0
        )
        reward -= swap_penalty
        info["swap_penalty"] = swap_penalty
        if truncated and self.forfeit_on_limit:
            # A deliberate safety-limit failure forfeits remaining lives. It is
            # an absorbing failure for training: no optimistic value bootstrap.
            # Replace the existing -1 shop penalty, but retain a final battle's
            # reward at a turn limit. Quitting cannot avoid future battle losses.
            penalty_already_paid = 1.0 if info.get("reason") == "shop_action_limit" else 0.0
            reward += penalty_already_paid - self.engine.state.lives
            terminated, truncated = True, False
            info["training_forfeit"] = True
        bonus = 0.0
        if (
            terminated
            and not truncated
            and self.engine.state.wins >= self.engine.config.target_wins
        ):
            bonus = max(
                0.0, self.success_bonus_max - self.success_action_cost * self.episode_actions
            )
            reward += bonus
        info["success_efficiency_bonus"] = bonus
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.engine.action_mask(), dtype=np.bool_)

    def _observation(self) -> Dict[str, np.ndarray]:
        state = self.engine.state
        config = self.engine.config
        previous = 0.0 if state.previous_outcome is None else (int(state.previous_outcome) + 1) / 2
        gold_bound = max(config.starting_gold * 2, 1)
        if self.engine.expanded_rules:
            # Five level-3 Swans can start with 25 gold. A fixed old 20-gold
            # clamp would hide spendable resources and break the Markov state.
            gold_bound = (
                config.starting_gold + config.max_team_size * 3 + config.max_actions_per_turn * 6
            )
        global_obs = np.asarray(
            [
                min(state.turn / config.max_turns, 1.0),
                min(state.gold / gold_bound, 1.0),
                min(state.lives / config.starting_lives, 1.0),
                min(state.wins / config.target_wins, 1.0),
                min(state.actions_this_turn / config.max_actions_per_turn, 1.0),
                previous,
                float(state.previous_outcome is not None),
                min(self.engine.unlocked_tier / 6, 1.0),
            ],
            dtype=np.float32,
        )
        if self.engine.midgame_rules:
            # Cake can accumulate sale value. Avoid clipping spendable resources
            # to a gold bound inherited from the old, Cake-free environment.
            global_obs[1] = state.gold / (state.gold + 10)
        if self.observe_episode_actions:
            # The success bonus depends on the whole episode, not just this shop.
            # Expose its sufficient statistic instead of hiding reward-relevant history.
            maximum = config.max_turns * config.max_actions_per_turn
            global_obs = np.append(global_obs, np.float32(min(self.episode_actions / maximum, 1.0)))
        if self.engine.tier4_rules:
            global_obs = np.append(
                global_obs,
                np.asarray(
                    [state.shop_attack_bonus / 50, state.shop_health_bonus / 50], dtype=np.float32
                ),
            )

        team = np.zeros(self.observation_space["team"].shape, dtype=np.float32)
        pet_count = len(self._pet_index)
        for row, pet in enumerate(state.team):
            team[row, 0] = 1.0
            team[row, 1 + self._pet_index[pet.spec_id]] = 1.0
            offset = 1 + pet_count
            team[row, offset : offset + 8] = np.asarray(
                [
                    min(pet.attack / 50, 1.0),
                    min(pet.health / 50, 1.0),
                    pet.level / 3,
                    pet.experience / 6,
                    min(pet.temporary_attack / 50, 1.0),
                    min(pet.temporary_health / 50, 1.0),
                    float(pet.perk == "honey"),
                    row / max(config.max_team_size - 1, 1),
                ],
                dtype=np.float32,
            )
            if self.engine.expanded_rules:
                team[row, offset + 8 : offset + 11] = [
                    float(pet.perk == "meat_bone"),
                    float(pet.perk == "melon"),
                    min(pet.ability_uses / 3, 1.0),
                ]
            if self.engine.midgame_rules:
                start = offset + 11
                team[row, start : start + 4] = [
                    float(pet.perk == "garlic"),
                    float(pet.perk == "chili"),
                    float(pet.perk == "cake"),
                    pet.sell_bonus / (pet.sell_bonus + 1),
                ]
                if pet.copied_ability is not None:
                    team[row, start + 4 + self._pet_index[pet.copied_ability]] = 1.0
            if self.engine.tier4_rules:
                team[row, -2:] = [float(pet.perk == "peanut"), float(pet.perk == "bread")]

        shop = np.zeros(self.observation_space["shop"].shape, dtype=np.float32)
        combined_count = len(self._pet_index) + len(self._food_index)
        for row, item in enumerate(state.shop):
            if item is None:
                continue
            shop[row, 0] = 1.0
            if item.kind == "pet":
                item_index = self._pet_index[item.item_id]
                tier = self.engine.catalog.pets[item.item_id].tier
                shop[row, 1 + combined_count + 1] = 1.0
            else:
                item_index = len(self._pet_index) + self._food_index[item.item_id]
                tier = self.engine.catalog.foods[item.item_id].tier
                shop[row, 1 + combined_count + 2] = 1.0
            shop[row, 1 + item_index] = 1.0
            offset = 1 + combined_count
            shop[row, offset] = min(item.cost / 10, 1.0)
            shop[row, offset + 3] = float(item.frozen)
            shop[row, offset + 4] = tier / 6
            if config.shop_pet_stats and item.kind == "pet":
                pet = item.pet
                if pet is None:
                    from .engine import make_pet

                    pet = make_pet(self.engine.catalog, item.item_id)
                shop[row, offset + 5 : offset + 11] = np.asarray(
                    [
                        min(pet.attack / 50, 1.0),
                        min(pet.health / 50, 1.0),
                        pet.experience / 6,
                        min(pet.temporary_attack / 50, 1.0),
                        min(pet.temporary_health / 50, 1.0),
                        float(pet.perk == "honey"),
                    ],
                    dtype=np.float32,
                )
            if self.engine.modern_shop and item.choice_group is not None:
                # Observe the actual pair relation, not an arbitrary increasing
                # group number. Buying this row removes every linked row.
                for other_row, other in enumerate(state.shop):
                    if other is not None and other.choice_group == item.choice_group:
                        shop[row, offset + 11 + other_row] = 1.0
            if self.engine.expanded_rules and item.kind == "pet" and item.pet is not None:
                start = offset + 11 + config.max_shop_size
                shop[row, start : start + 3] = [
                    float(item.pet.perk == "meat_bone"),
                    float(item.pet.perk == "melon"),
                    min(item.pet.ability_uses / 3, 1.0),
                ]
                if self.engine.midgame_rules:
                    start += 3
                    pet = item.pet
                    shop[row, start : start + 4] = [
                        float(pet.perk == "garlic"),
                        float(pet.perk == "chili"),
                        float(pet.perk == "cake"),
                        pet.sell_bonus / (pet.sell_bonus + 1),
                    ]
                    if pet.copied_ability is not None:
                        shop[row, start + 4 + self._pet_index[pet.copied_ability]] = 1.0
                if self.engine.tier4_rules:
                    shop[row, -2:] = [
                        float(item.pet.perk == "peanut"),
                        float(item.pet.perk == "bread"),
                    ]

        return {"global": global_obs, "team": team, "shop": shop}

    def render(self) -> Optional[str]:
        state = self.engine.state
        team = (
            " | ".join(
                f"{self.engine.catalog.pets[pet.spec_id].name} "
                f"{pet.attack}/{pet.health} L{pet.level}"
                for pet in state.team
            )
            or "(empty)"
        )
        shop = " | ".join(
            "(empty)"
            if item is None
            else f"{'*' if item.frozen else ''}{item.item_id}({item.cost})"
            for item in state.shop
        )
        view = (
            f"Turn {state.turn}  gold={state.gold} lives={state.lives} wins={state.wins}\n"
            f"Team: {team}\nShop: {shop}"
        )
        if self.render_mode == "human":
            print(view)
            return None
        return view
