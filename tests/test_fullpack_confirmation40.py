import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fullpack_confirmation_test", ROOT / "scripts/confirm_fullpack_tier6.py"
)
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def evidence(wins=400, forced=0):
    return {
        str(model): {
            "families": {
                family: {
                    "episodes": 500,
                    "deterministic": True,
                    "episode_results": [
                        {
                            "seed": run.SEED + i,
                            "success": i < wins,
                            "forced_end_turns": int(i < forced),
                            "truncated": False,
                            "wins": 10 if i < wins else 7,
                            "return": 5.0,
                            "action_counts": {"buy_pet": 10},
                        }
                        for i in range(500)
                    ],
                }
                for family in run.FAMILIES
            }
        }
        for model in run.campaign.SEEDS
    }


@pytest.mark.parametrize("wins,forced,passed", [(400, 25, True), (399, 0, False), (450, 26, False)])
def test_exact_rate_and_per_family_reliability(wins, forced, passed):
    result = run.summarize(evidence(wins, forced), run.SEED)
    assert result["passed"] is passed
    assert result["learned_episodes"] == 3000
    assert result["learned_successes"] == wins * 6


def test_missing_model_cannot_be_averaged_away():
    result = evidence()
    result.pop(str(run.campaign.SEEDS[0]))
    with pytest.raises(ValueError, match="three models"):
        run.summarize(result, run.SEED)


def test_duplicate_or_screening_seeds_are_rejected():
    result = evidence()
    first = next(iter(result.values()))
    next(iter(first["families"].values()))["episode_results"][0]["seed"] = 120_000_000
    with pytest.raises(ValueError, match="episode seeds"):
        run.summarize(result, run.SEED)


def test_scripted_family_cannot_be_omitted():
    result = evidence()
    next(iter(result.values()))["families"].pop("full_stats")
    with pytest.raises(ValueError, match="five frozen test families"):
        run.summarize(result, run.SEED)


def test_one_truncated_episode_fails_confirmation():
    result = evidence()
    next(iter(next(iter(result.values()))["families"].values()))["episode_results"][0][
        "truncated"
    ] = True
    assert not run.summarize(result, run.SEED)["passed"]
