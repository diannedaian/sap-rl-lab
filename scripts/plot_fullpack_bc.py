"""Plot recorded fullpack opponent BC histories; these are not candidate win rates."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from sap_rl_lab.fullpack.evaluation import file_digest
from sap_rl_lab.fullpack.ppo_guardrails import write_new


def plot(root, output):
    paths = sorted((root / "bc").glob("*/fit.json"))
    if len(paths) != 7:
        raise ValueError("Expected all seven independent opponent BC histories")
    histories = {p.parent.name: json.loads(p.read_text())["history"] for p in paths}
    for h in histories.values():
        if [r["epoch"] for r in h] != list(range(21)):
            raise ValueError("Expected initialization and twenty BC epochs")
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ("cross_entropy", "action_accuracy"),
        ("Held-out teacher-action cross entropy", "Held-out teacher-action accuracy"),
    ):
        values = np.array([[r["validation"][key] for r in h] for h in histories.values()])
        for row in values:
            ax.plot(range(21), row, color="#94a3b8", alpha=0.5, linewidth=1)
        ax.plot(range(21), values.mean(axis=0), color="#2563eb", label="Seven-model mean")
        ax.plot([], [], color="#94a3b8", label="Individual BC runs")
        ax.set_title(title)
        ax.set_xlabel("BC epoch (zero = random initialization)")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend()
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylim(0, 1)
    fig.suptitle("Fullpack opponent generators: behavior cloning, not candidate PPO win rates")
    fig.savefig(output / "bc.png", dpi=160)
    plt.close(fig)
    write_new(
        output / "evidence.json",
        {
            "histories": histories,
            "input_sha256": {str(p): file_digest(p) for p in paths},
            "script_sha256": file_digest(__file__),
            "scope": "Seven opponent BC initializations; candidates inherit the Tier5 curriculum. "
            "Imitating heuristic teacher actions is not a measure of game strength.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.campaign.resolve(), args.output.resolve())
