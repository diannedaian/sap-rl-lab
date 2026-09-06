from sap_rl_lab.round5 import ARMS, SEEDS, aggregate, select_delivery


def test_delivery_selection_uses_validation_not_test_or_cutoffs():
    candidates = {
        "swap_cost-seed503": {"validation_success": 0.9, "validation_return": 7, "seed": 503},
        "swap_cost-seed607": {"validation_success": 0.9, "validation_return": 7, "seed": 607},
        "swap_cost-seed709": {"validation_success": 0.8, "validation_return": 9, "seed": 709},
        "unshaped-seed503": {"validation_success": 1, "validation_return": 10, "seed": 503},
    }
    assert select_delivery(candidates) == "swap_cost-seed503"
    candidates["swap_cost-seed607"]["validation_return"] = 8
    assert select_delivery(candidates) == "swap_cost-seed607"


def test_closure_screen_rejects_one_unreliable_seed():
    results = {
        f"{arm}-seed{seed}": {
            "success_rate": 0.9,
            "truncation_rate": 0.005 if arm == "swap_cost" else 0.03,
            "mean_episode_actions": 80,
            "mean_return": 7,
        }
        for arm in ARMS
        for seed in SEEDS
    }
    assert aggregate(results)["closure_screen_passed"]
    results["swap_cost-seed709"]["truncation_rate"] = 0.011
    outcome = aggregate(results)
    assert not outcome["closure_screen_passed"]
    assert outcome["predeclared_checks"]["mean_cutoffs_lower_than_equal_budget_control"]
