"""Plot frozen Round 3 validation and per-family test results, without reselection."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("output_directory")
    args = parser.parse_args()
    root, output = Path(args.run_directory), Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    colors = {"single": "#2563eb", "mixed": "#d97706"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for arm, color in colors.items():
        for i, seed in enumerate(protocol["seeds"]):
            history = json.loads((root / f"{arm}-seed{seed}/validation_history.json").read_text())
            axes[0].plot(
                [h["timesteps"] / 1e6 for h in history],
                [h["success_rate"] for h in history],
                color=color,
                alpha=0.75,
                label=f"{arm.title()} opponents" if i == 0 else None,
            )
    families = ["greedy", "stats", "summon", "round2_challenge"]
    for x, family in enumerate(families):
        for arm, color in colors.items():
            offset = -0.12 if arm == "single" else 0.12
            values = [
                summary["results"][f"{arm}-seed{s}"]["families"][family]["success_rate"]
                for s in protocol["seeds"]
            ]
            axes[1].scatter(
                [x + offset + (i - 1) * 0.025 for i in range(len(values))],
                values,
                color=color,
                s=40,
                alpha=0.8,
                label=f"{arm.title()} opponents" if x == 0 else None,
            )
        for name, marker, color in (("round2", "D", "#475569"), ("greedy", "x", "#16a34a")):
            value = summary["results"][name]["families"][family]["success_rate"]
            axes[1].scatter(
                [x],
                [value],
                marker=marker,
                color=color,
                s=52,
                label=f"{name.title()} reference" if x == 0 else None,
            )
    axes[0].set(
        title="Validation: equal-family average",
        xlabel="Training decisions (millions)",
        ylabel="10-win run success",
    )
    axes[1].set(title="Fresh test: each dot is one training seed")
    axes[1].set_xticks(range(4), ["Spend-gold", "Stats", "Summon", "Frozen R2\n(test only)"])
    axes[1].axvline(2.5, color="#94a3b8", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_ylim(-0.025, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.suptitle("SAP RL Lab · Round 3: opponent diversity from scratch", fontsize=16)
    fig.supxlabel(
        "Three independent initializations, paired between arms. "
        "The frozen-agent challenge is excluded from the primary average.",
        fontsize=9,
    )
    fig.savefig(output / "comparison.png", dpi=170)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    with (output / "comparison.csv").open("x") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "policy",
                "family",
                "success_rate",
                "mean_wins",
                "mean_return",
                "cutoff_rate",
                "mean_actions",
            ]
        )
        for name, result in summary["results"].items():
            for family, metrics in result["families"].items():
                writer.writerow(
                    [
                        name,
                        family,
                        metrics["success_rate"],
                        metrics["mean_wins"],
                        metrics["mean_return"],
                        metrics["truncation_rate"],
                        metrics["mean_episode_actions"],
                    ]
                )
    print(output / "comparison.png")


if __name__ == "__main__":
    main()
