"""Numerical stopping screen for expanded confirmation, not a parity certificate.

Consumes full evaluate_suite results for paired candidate/control models.
The caller must separately freeze selections, verify provenance and keep tests
held out. Passing this function does not establish any of those facts.
"""

from fractions import Fraction

FAMILIES = {"expanded_stats", "expanded_summon", "expanded_tempo"}


def screen(pairs):
    """Recompute gates from raw episode rows, never rounded macro headlines."""
    if len(pairs) != 3 or len({pair["seed"] for pair in pairs}) != 3:
        raise ValueError("Confirmation requires exactly three distinct training seeds")
    per_seed = []
    for pair in pairs:
        candidate, control = pair["candidate"], pair["control"]
        if set(candidate["families"]) != FAMILIES or set(control["families"]) != FAMILIES:
            raise ValueError("Confirmation requires all three expanded scripted families")
        families = {}
        exact_rates = []
        for family in sorted(FAMILIES):
            left, right = candidate["families"][family], control["families"][family]
            a, b = left["episode_results"], right["episode_results"]
            if len(a) < 1000 or len(a) != len(b):
                raise ValueError("Each paired family requires at least 1000 full episode rows")
            seeds_a, seeds_b = [r["seed"] for r in a], [r["seed"] for r in b]
            if seeds_a != seeds_b or len(set(seeds_a)) != len(seeds_a):
                raise ValueError("Episode seeds must be unique and paired in order")
            if not left["league_sha256"] or left["league_sha256"] != right["league_sha256"]:
                raise ValueError("Candidate/control must use identical frozen opponent pools")
            if left["environment_contract"] != right["environment_contract"]:
                raise ValueError("Candidate/control environment contracts differ")
            if (
                left["environment_contract"]["game_config"]["shop_action_limit_mode"]
                != "force_battle"
            ):
                raise ValueError("This screen requires the force-battle contract")
            for row in a + b:
                if not 0 <= row["forced_end_turns"] <= row["battles"]:
                    raise ValueError("Invalid forced-battle count")
                if type(row["success"]) is not bool or type(row["truncated"]) is not bool:
                    raise ValueError("Success and truncation flags must be boolean")
            forced = sum(r["forced_end_turns"] for r in a)
            battles = sum(r["battles"] for r in a)
            success = Fraction(sum(r["success"] for r in a), len(a))
            control_success = Fraction(sum(r["success"] for r in b), len(b))
            truncation = Fraction(sum(r["truncated"] for r in a), len(a))
            forcing = Fraction(sum(r["forced_end_turns"] > 0 for r in a), len(a))
            exact_rates.append((success, control_success, truncation, forcing))
            families[family] = {
                "episodes": len(a),
                "success_rate": float(success),
                "control_success_rate": float(control_success),
                "truncation_rate": float(truncation),
                "forced_episode_rate": float(forcing),
                "forced_battle_rate": forced / max(battles, 1),
            }
        # Compare exact counts, not float means: both 90% and a 1pp loss are
        # inclusive boundaries. The macro remains an unweighted family mean.
        macro_success = sum(r[0] for r in exact_rates) / len(exact_rates)
        macro_control = sum(r[1] for r in exact_rates) / len(exact_rates)
        per_seed.append(
            {
                "seed": pair["seed"],
                "families": families,
                "macro_success": float(macro_success),
                "macro_control_success": float(macro_control),
                "checks": {
                    "every_family_truncation_below_1pct": all(
                        r[2] < Fraction(1, 100) for r in exact_rates
                    ),
                    "every_family_forcing_below_1pct": all(
                        r[3] < Fraction(1, 100) for r in exact_rates
                    ),
                    "macro_success_at_least_90pct": macro_success >= Fraction(9, 10),
                    "macro_loss_vs_control_at_most_1pp": (
                        macro_control - macro_success <= Fraction(1, 100)
                    ),
                },
            }
        )
    return {
        "per_seed": per_seed,
        "numerical_screen_passed": all(all(row["checks"].values()) for row in per_seed),
        "goal_complete": False,
        "remaining_evidence": "Frozen pre-test selection, source/data/model provenance, "
        "independent runs, rule parity, unseen learned-opponent challenge, and replay inspection "
        "must be verified separately. This is an engineering screen, not a significance test.",
    }
