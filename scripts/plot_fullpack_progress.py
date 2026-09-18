"""Read-only raw validation curves across full Tier6 continuation segments.

Recount episode rows and replay the frozen selector. No smoothing, test queries,
training, or replacement of checkpoints. Zero means the transferred Tier5 model.
"""

import argparse
import json
from pathlib import Path

from sap_rl_lab.fullpack.evaluation import file_digest
from sap_rl_lab.fullpack.ppo_guardrails import GuardrailConfig, write_new
from sap_rl_lab.fullpack.recipe import LearnedFirstGuard

SEEDS = (85101, 85201, 85301)


def extract(root):
    inputs, curves, cases = {}, {}, {}

    def read(path):
        text = path.read_bytes()
        data = json.loads(text)
        import hashlib

        inputs[str(path)] = hashlib.sha256(text).hexdigest()
        return data

    for seed in SEEDS:
        guard = LearnedFirstGuard(GuardrailConfig(stop_on_regression=False))
        points, offset = [], 0
        for segment in sorted((root / "segments").glob("segment*")):
            directory = segment / f"candidate-{seed}"
            if not directory.exists():
                break
            for path in sorted(directory.glob("eval*.json")):
                record = read(path)
                checkpoint, result = record["checkpoint"], record["evaluation"]
                if file_digest(checkpoint["path"]) != checkpoint["sha256"]:
                    raise ValueError("Validation checkpoint changed")
                inputs[checkpoint["path"]] = checkpoint["sha256"]
                if len(result["families"]) != 5:
                    raise ValueError("Expected all five validation families")
                for family, evidence in result["families"].items():
                    rows = evidence["episode_results"]
                    if len(rows) != 200 or [r["seed"] for r in rows] != list(
                        range(119000000, 119000200)
                    ):
                        raise ValueError("Validation episode set changed")
                    identity = evidence["league_sha256"]
                    if identity != cases.setdefault(family, identity):
                        raise ValueError("Validation opponent pool changed across checkpoints")
                decision = guard.consider(result, checkpoint)
                points.append(
                    {
                        "segment": segment.name,
                        "fullpack_steps": offset + checkpoint["timesteps"],
                        "learned_success": decision["score"][0],
                        "eligible": decision["eligible"],
                        "selected_best": guard.best["score"][0] if guard.best else None,
                        "max_family_forced_rate": max(
                            v["forcing"] for v in decision["families"].values()
                        ),
                        "max_family_truncation": max(
                            v["truncation"] for v in decision["families"].values()
                        ),
                        "checkpoint": checkpoint,
                    }
                )
            complete = directory / "complete.json"
            if not complete.exists():
                break
            offset += read(complete)["actual_timesteps"]
        curves[str(seed)] = points
    if not any(curves.values()):
        raise ValueError("No completed validation records")
    return {
        "curves": curves,
        "input_sha256": inputs,
        "scope": "60 pets; full Tier1-6 shops; 40-turn cap; validation only",
        "x_axis": "Additional full Tier6 decisions; excludes Tier5 and opponent training",
        "notes": [
            "Raw fixed-seed validation rates; two learned families x200 episodes per point.",
            "Solid = current checkpoint; dashed = best eligible validation selection so far.",
            "Forced rate = highest of five family rates, not an average over families.",
            "Different lineages can have different completed training steps in a live snapshot.",
            "This is not an independent test or evidence that the 80% goal is complete.",
        ],
    }


def plot(root, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    data = extract(root)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "evidence.json", data)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for seed, label, color in zip(SEEDS, "ABC", ("#2563eb", "#d97706", "#059669")):
        points = data["curves"][str(seed)]
        if not points:
            continue
        x = [p["fullpack_steps"] / 1e6 for p in points]
        axes[0].plot(x, [p["learned_success"] for p in points], "o-", color=color, label=label)
        axes[0].plot(x, [p["selected_best"] for p in points], "--", color=color, alpha=0.7)
        axes[1].plot(x, [p["max_family_forced_rate"] for p in points], "o-", color=color)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].axhline(0.8, color="#64748b", linestyle=":", label="80% reference")
    axes[0].plot([], [], "--", color="#64748b", label="Best eligible so far")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Learned-opponent 10-win rate")
    axes[0].set_title("Full Turtle Pack validation: raw checkpoints and best eligible so far")
    axes[0].legend(ncol=3, loc="lower right")
    axes[1].axhline(0.05, color="#64748b", linestyle=":", label="5% delivery gate")
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Worst-family forced-episode rate")
    axes[1].set_xlabel("Full Tier6 training decisions (millions; zero = warm-started model)")
    axes[1].legend()
    fig.savefig(output / "validation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v[-1] if v else None for k, v in data["curves"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.campaign.resolve(), args.output.resolve())
