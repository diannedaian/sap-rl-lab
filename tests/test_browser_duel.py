"""Inference parity and head-to-head integration; no original rules are modified."""

import hashlib
import json
import random
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
MaskablePPO = pytest.importorskip("sb3_contrib").MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "viewer/duel"
sys.path.insert(0, str(ASSETS / "runtime.zip"))
from sap_web.duel import Duel, NumpyPolicy, ObservedTrace  # noqa: E402
from sap_web.engine import AutoBattler, resolve_battle  # noqa: E402
from sap_web.late_events import late_battle_runtime  # noqa: E402

from sap_rl_lab.fullpack.env import SapAutoBattlerEnv  # noqa: E402
from sap_rl_lab.fullpack.evaluation import policy_environment_options  # noqa: E402


@pytest.fixture(scope="module")
def policy():
    torch.set_num_threads(1)
    return NumpyPolicy(ASSETS)


@pytest.fixture(scope="module")
def torch_model():
    import os

    supplied = os.environ.get("SAP_RL_CHECKPOINT")
    selection_path = ROOT / "runs/fullpack-tier6-h40-v1/segments/segment005/selection.json"
    if supplied:
        path = Path(supplied)
    elif selection_path.exists():
        selection = json.loads(selection_path.read_text())["selected"]["85101"]["checkpoint"]
        path = Path(selection["path"])
    else:
        pytest.skip("Original Torch checkpoint is optional; set SAP_RL_CHECKPOINT to run parity")
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["checkpoint_sha256"]
    return MaskablePPO.load(path, device="cpu")


def test_export_is_selected_model_and_exact_rule_copy(policy):
    assert policy.manifest["parameters"] == 341516
    assert len(policy.catalog.rollable_pet_ids) == 60
    assert (policy.config.max_turns, policy.config.max_shop_tier) == (40, 6)
    for name, expected in policy.manifest["files"].items():
        assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == expected
    with zipfile.ZipFile(ASSETS / "runtime.zip") as archive:
        for name, expected in policy.manifest["rule_sha256"].items():
            original = (ROOT / f"src/sap_rl_lab/fullpack/{name}.py").read_bytes()
            assert archive.read(f"sap_web/{name}.py") == original
            assert hashlib.sha256(original).hexdigest() == expected


def test_numpy_matches_torch_on_1200_live_states(policy, torch_model):
    env = SapAutoBattlerEnv(**policy_environment_options(torch_model))
    obs, _ = env.reset(seed=20260918)
    rng = random.Random(4412)
    max_probability_error, max_value_error = 0.0, 0.0
    for i in range(1200):
        np_obs = policy.observe(env.engine)
        for key in obs:
            np.testing.assert_array_equal(obs[key], np_obs[key])
        mask = env.action_masks()
        action, probabilities, value = policy.forward(np_obs, mask)
        with torch.no_grad():
            tensor, _ = torch_model.policy.obs_to_tensor(obs)
            distribution = torch_model.policy.get_distribution(tensor, action_masks=mask)
            reference_probs = distribution.distribution.probs.cpu().numpy().ravel()
            reference_value = float(torch_model.policy.predict_values(tensor).item())
        max_probability_error = max(
            max_probability_error, float(np.max(abs(probabilities - reference_probs)))
        )
        max_value_error = max(max_value_error, abs(value - reference_value))
        np.testing.assert_allclose(probabilities, reference_probs, atol=1e-5, rtol=2e-4)
        assert value == pytest.approx(reference_value, abs=2e-5)
        assert action == int(np.argmax(reference_probs))
        # Predominantly on-policy with some legal perturbations.
        chosen = rng.choice(np.flatnonzero(mask).tolist()) if i % 7 == 0 else action
        obs, _, done, cutoff, _ = env.step(chosen)
        if done or cutoff:
            obs, _ = env.reset(seed=20260918 + i + 1)
    print(
        f"parity: 1200/1200 actions; max probability error={max_probability_error:.3g}; "
        f"value error={max_value_error:.3g}"
    )


def test_shop_actions_freeze_sell_and_stale_requests(policy):
    duel = Duel(policy, 123)
    before = duel.view()
    buy = next(a for a in before["legal"] if a["kind"] == "buy_pet")
    duel.act(buy["id"], 0)
    assert len(duel.human.state.team) == 1
    assert duel.human.state.gold == 7
    after = deepcopy(duel.view())
    with pytest.raises(ValueError):
        duel.act(buy["id"], 0)
    assert duel.view() == after
    with pytest.raises(ValueError):
        duel.act(999, duel.revision)
    assert duel.view() == after
    freeze = next(a for a in duel.view()["legal"] if a["kind"] == "toggle_freeze")
    item = duel.human.state.shop[freeze["source"]]
    duel.act(freeze["id"], duel.revision)
    assert item.frozen
    duel.act(1, duel.revision)
    # The frozen engine compacts frozen items toward the relevant shop edge.
    assert item in duel.human.state.shop and item.frozen


def test_forced_battle_once_and_one_shared_result(policy):
    duel = Duel(policy, 22)
    for _ in range(30):
        freeze = next(a for a in duel.view()["legal"] if a["kind"] == "toggle_freeze")
        duel.act(freeze["id"], duel.revision)
    assert duel.phase == "battle"
    assert duel.battle["human_forced"]
    assert duel.human.state.turn == duel.ai.state.turn == 2
    assert int(duel.human.state.previous_outcome) == -int(duel.ai.state.previous_outcome)
    assert duel.human.state.lives == 4 and duel.ai.state.wins == 1
    assert duel.view()["legal"] == []
    with pytest.raises(ValueError):
        duel.act(0, duel.revision)
    duel.next_round(duel.revision)
    assert duel.human.state.actions_this_turn == 0
    assert duel.human.state.gold == 10


def test_empty_player_complete_match_and_no_ai_shop_leak(policy):
    duel = Duel(policy, 99)
    assert "shop" not in duel.view()["ai"]
    assert duel.view()["ai"]["team"] == []
    for _ in range(40):
        duel.act(0, duel.revision)
        if duel.done:
            break
        duel.next_round(duel.revision)
    assert duel.done and duel.result == "ai"
    assert duel.human.state.lives == 0
    with pytest.raises(ValueError):
        duel.next_round(duel.revision)


@pytest.mark.parametrize("seed", [4, 10, 371])
def test_complete_self_play_is_deterministic(policy, seed):
    def run():
        duel = Duel(policy, seed)
        while not duel.done:
            while duel.phase == "shop":
                action, _, _ = policy.predict(duel.human)
                duel.act(action, duel.revision)
            if not duel.done:
                duel.next_round(duel.revision)
        return duel.view()

    assert run() == run()


def test_observer_and_settlement_match_frozen_engine(policy):
    # Late-game duel supplies realistic perk/ability states for independent parity checks.
    duel = Duel(policy, 712)
    checked = 0
    while not duel.done:
        before = deepcopy(duel.human)
        while duel.phase == "shop":
            before = deepcopy(duel.human)
            action, _, _ = policy.predict(duel.human)
            duel.act(action, duel.revision)
        from sap_web.domain import Pet

        enemy = [Pet(**p) for p in duel.battle["ai_team"]]
        reference = AutoBattler(
            policy.catalog, policy.config, lambda *args, team=enemy: deepcopy(team)
        )
        reference.state, reference.rng = before.state, before.rng
        # The duel uses an independent combat RNG; fixed result injects only that draw.
        import sap_web.engine as engine_module

        old_resolver = engine_module.resolve_battle
        try:
            engine_module.resolve_battle = lambda *args, **kwargs: duel.human.last_battle
            reference.step_id(action)
        finally:
            engine_module.resolve_battle = old_resolver
        assert reference.state == duel.human.state
        assert reference.rng.getstate() == duel.human.rng.getstate()
        friendly = [Pet(**p) for p in duel.battle["human_team"]]
        a_rng, b_rng = random.Random(checked), random.Random(checked)
        standard = resolve_battle(friendly, enemy, policy.catalog, a_rng)
        runtime = late_battle_runtime(friendly, enemy, policy.catalog, b_rng)
        frames = []
        runtime.trace = ObservedTrace(runtime, frames)
        outcome, attacks = runtime.battle()
        assert (outcome, attacks, list(runtime.trace)) == (
            standard.outcome,
            standard.attacks,
            standard.trace,
        )
        assert a_rng.getstate() == b_rng.getstate()
        checked += 1
        if not duel.done:
            duel.next_round(duel.revision)
    assert checked >= 5
