"""Render a completed comparison using its saved validation and test evidence."""

import argparse
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
    root = Path(args.run_directory)
    summary = json.loads((root / "summary.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    colors = {"control": "#2563eb", "efficient": "#d56a25"}
    labels = {"control": "Longer training", "efficient": "+ Action/forfeit penalties"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), layout="constrained")
    for arm in colors:
        for index, seed in enumerate(protocol["continuation_seeds"]):
            history = json.loads((root / f"{arm}-seed{seed}/validation_history.json").read_text())
            axes[0].plot(
                [row["timesteps"] / 1_000_000 for row in history],
                [row["success_rate"] for row in history],
                color=colors[arm],
                alpha=0.7,
                linewidth=1.6,
                label=labels[arm] if index == 0 else None,
            )
        values = [
            summary["results"][f"{arm}-seed{s}"]["success_rate"]
            for s in protocol["continuation_seeds"]
        ]
        offset = -0.12 if arm == "control" else 0.12
        positions = [i + offset for i in range(len(values))]
        axes[1].scatter(positions, values, color=colors[arm], s=75, zorder=5, label=labels[arm])
        for x, y in zip(positions, values):
            axes[1].annotate(
                f"{y:.1%}",
                (x, y),
                xytext=(0, 9 if offset > 0 else -17),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
    for name, color, style in (("original", "#475569", "--"), ("greedy", "#94a3b8", ":")):
        value = summary["results"][name]["success_rate"]
        axes[1].axhline(
            value,
            color=color,
            linestyle=style,
            linewidth=1.5,
            label=f"{name.title()} ({value:.1%})",
        )
    axes[0].set(
        title="Validation during training",
        xlabel="Additional decisions (millions)",
        ylabel="10-win run success",
    )
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[1].set(title="Fresh test: 1,000 episodes per policy", xlabel="Continuation seed")
    axes[1].set_xticks(range(len(protocol["continuation_seeds"])), protocol["continuation_seeds"])
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16)
    fig.suptitle("SAP RL Lab · Round 2", fontsize=17, fontweight="bold")
    fig.supxlabel(
        "Three continuation seeds share one initial policy; all test pools/seeds match.",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.svg")
    plt.close(fig)
    rows = ["policy,success_rate,mean_wins,mean_return,truncation_rate,mean_actions,swap_fraction"]
    for name, result in summary["results"].items():
        swap_fraction = result["action_counts"].get("swap_adjacent", 0) / sum(
            result["action_counts"].values()
        )
        rows.append(
            f"{name},{result['success_rate']},{result['mean_wins']},{result['mean_return']},"
            f"{result['truncation_rate']},{result['mean_episode_actions']},{swap_fraction}"
        )
    (out / "comparison.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(out / "comparison.png")


if __name__ == "__main__":
    main()
