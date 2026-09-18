"""Exact-count audit of already completed confirmation results, never training.

The frozen screen still supplies structural validation. Rational comparisons
replace only its numeric decisions, preserving all predeclared thresholds.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded import verify_inputs
from sap_rl_lab.expanded_gate import screen


def exact_screen(pairs):
    report = screen(pairs)
    original = [dict(item["checks"]) for item in report["per_seed"]]
    for pair, result in zip(pairs, report["per_seed"]):
        rates = []
        for family in sorted(pair["candidate"]["families"]):
            a = pair["candidate"]["families"][family]["episode_results"]
            b = pair["control"]["families"][family]["episode_results"]
            rates.append(
                {
                    "candidate": Fraction(sum(row["success"] for row in a), len(a)),
                    "control": Fraction(sum(row["success"] for row in b), len(b)),
                    "forcing": Fraction(sum(row["forced_end_turns"] > 0 for row in a), len(a)),
                    "truncation": Fraction(sum(row["truncated"] for row in a), len(a)),
                }
            )
        candidate = sum(row["candidate"] for row in rates) / len(rates)
        control = sum(row["control"] for row in rates) / len(rates)
        result["checks"] = {
            "every_family_truncation_below_1pct": all(
                row["truncation"] < Fraction(1, 100) for row in rates
            ),
            "every_family_forcing_below_1pct": all(
                row["forcing"] < Fraction(1, 100) for row in rates
            ),
            "macro_success_at_least_90pct": candidate >= Fraction(9, 10),
            "macro_loss_vs_control_at_most_1pp": control - candidate <= Fraction(1, 100),
        }
        result["exact_macro_success"] = str(candidate)
        result["exact_macro_control_success"] = str(control)
    report["numerical_screen_passed"] = all(
        all(row["checks"].values()) for row in report["per_seed"]
    )
    report["numeric_method"] = "Exact rational per-family rates; macro is unweighted family mean"
    report["frozen_screen_checks"] = original
    report["numeric_decision_changed"] = any(
        old != new["checks"] for old, new in zip(original, report["per_seed"])
    )
    return report


def audit(confirmation, output):
    confirmation, output = Path(confirmation).resolve(), Path(output).resolve()

    def read(path):
        return json.loads(path.read_text())

    protocol = read(confirmation / "protocol.json")
    verify_inputs(confirmation, protocol)
    selection = read(confirmation / "selection.json")
    original_summary = read(confirmation / "summary.json")
    if (
        selection["protocol_sha256"] != file_digest(confirmation / "protocol.json")
        or original_summary["selection_sha256"] != file_digest(confirmation / "selection.json")
    ):
        raise ValueError("Requires an unchanged, completed frozen confirmation summary")
    pairs, inputs = [], {}
    for seed in protocol["seeds"]:
        pair = {"seed": seed}
        for role, arm in (("candidate", protocol["candidate_arm"]), ("control", "control")):
            name = f"{arm}-seed{seed}"
            path = confirmation / f"evaluation-{name}.json"
            pair[role] = read(path)["test"]
            inputs[path.name] = file_digest(path)
        pairs.append(pair)
    report = exact_screen(pairs)
    report["provenance"] = {
        "summary_sha256": file_digest(confirmation / "summary.json"),
        "selection_sha256": file_digest(confirmation / "selection.json"),
        "evaluation_sha256": inputs,
        "audit_script_sha256": file_digest(__file__),
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(
        json.dumps({k: report[k] for k in ("numerical_screen_passed", "numeric_decision_changed")})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.confirmation, args.output)
