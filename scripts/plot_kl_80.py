"""Plot the frozen KL-80 validation records; never run or reselect a policy."""

import argparse
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

STYLES = [("#2864B7", "o", "-"), ("#CE6B20", "s", "--"), ("#168269", "^", "-.")]
SCRIPTED = ("midgame_stats", "midgame_summon", "midgame_tempo")
HARD = "validation_balanced"


def extract(root):
    """Recount all plotted rates from episode rows and preserve input hashes."""
    hashes, curves, cases = {}, {}, {}

    def read(path):
        payload = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    protocol = read(root / "protocol.json")
    for arm, config in protocol["arms"].items():
        complete = read(root / arm / "complete.json")
        points = []
        training = config["training"]
        for entry in complete["evaluations"]:
            checkpoint = entry["checkpoint"]
            source = root / arm / Path(checkpoint["path"]).with_suffix(".json").name
            record = read(source)
            assert record["checkpoint"] == checkpoint
            families = record["evaluation"]["families"]
            assert set(families) == {*SCRIPTED, HARD}
            metrics = {}
            for family, evaluation in families.items():
                rows = evaluation["episode_results"]
                n = len(rows)
                assert n == evaluation["episodes"] == training["validation_episodes"]
                assert evaluation["seed_start"] == training["validation_seed"]
                identity = (evaluation["league_sha256"], tuple(row["seed"] for row in rows))
                assert identity == cases.setdefault(family, identity)
                successes = sum(bool(row["success"]) for row in rows)
                forced = sum(row["forced_end_turns"] > 0 for row in rows)
                decision = entry["decision"]["families"][family]
                assert isclose(successes / n, decision["success"])
                assert isclose(forced / n, decision["forcing"])
                metrics[family] = {"episodes": n, "successes": successes, "forced": forced}
            macro = fmean(m["successes"] / m["episodes"] for m in metrics.values())
            assert isclose(macro, entry["decision"]["score"][0])
            points.append(
                {
                    "steps": checkpoint["timesteps"],
                    "macro_success": macro,
                    "hard_success": metrics[HARD]["successes"] / metrics[HARD]["episodes"],
                    "scripted_success": fmean(
                        metrics[f]["successes"] / metrics[f]["episodes"] for f in SCRIPTED
                    ),
                    "forced_episode_rate": sum(m["forced"] for m in metrics.values())
                    / sum(m["episodes"] for m in metrics.values()),
                    "family_counts": metrics,
                }
            )
        expected = list(range(0, training["timesteps"] + 1, training["evaluation_interval"]))
        assert [p["steps"] for p in points] == expected
        curves[arm] = {
            "source_seed": config["source_seed"],
            "continuation_rng_seed": training["seed"],
            "selected_steps": complete["selected"]["checkpoint"]["timesteps"],
            "points": points,
        }
    review = read(root / "final_review.json")
    return {
        "run_directory": str(root),
        "curves": curves,
        "formal_test_endpoints": review["results"],
        "input_sha256": hashes,
        "notes": [
            "Validation only: fixed cases at each checkpoint, raw unsmoothed rates.",
            "Each point: four families x 100 episodes; validation_balanced alone has 100.",
            "10-win episode success is not single-battle win rate.",
            "Forced episode rate counts episodes with at least one forced battle, not turns.",
            "X is additional decisions this round; zero is already a trained policy.",
            "Source-seed labels identify lineages, not fresh continuation RNG seeds.",
            "Formal learned-opponent test is different; endpoints are not a dense curve.",
            "No smoothing, test-based checkpoint reselection, new training or evaluation.",
        ],
    }


def draw(data, output, font_path):
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "axes.unicode_minus": False,
            "text.color": "#253040",
            "axes.labelcolor": "#253040",
            "xtick.color": "#455163",
            "ytick.color": "#455163",
            "svg.fonttype": "path",
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.4, 10.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15, 0.52]},
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.85, bottom=0.12, hspace=0.36)
    handles = []
    for curve, (color, marker, style) in zip(data["curves"].values(), STYLES):
        points = curve["points"]
        x = [p["steps"] / 1e6 for p in points]
        chosen = "选起点" if curve["selected_steps"] == 0 else "选终点"
        label = f"{curve['source_seed']}（{chosen}）"
        for index, field in enumerate(("macro_success", "hard_success", "forced_episode_rate")):
            (line,) = axes[index].plot(
                x,
                [p[field] for p in points],
                color=color,
                marker=marker,
                linestyle=style,
                linewidth=2,
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=1.6,
                label=label,
            )
            if index == 0:
                handles.append(line)
    titles = [
        "A  总体验证：四类对手平均（每点 400 局）",
        "B  单看学习型验证对手：validation_balanced（每点 100 局）",
        "C  强制开战：一局里至少触发一次（每点 400 局）",
    ]
    for index, ax in enumerate(axes):
        ax.set_title(titles[index], loc="left", pad=11)
        ax.set_ylabel("十胜率" if index < 2 else "强制局率")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#ADB7C3")
        ax.grid(axis="y", color="#DDE2E8", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1 if index == 2 else 0))
    axes[0].set_ylim(0.90, 1.005)
    axes[0].set_yticks([0.90, 0.925, 0.95, 0.975, 1.00])
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    axes[1].set_ylim(0.67, 0.925)
    axes[1].set_yticks([0.70, 0.75, 0.80, 0.85, 0.90])
    axes[2].set_ylim(-0.0006, 0.006)
    axes[2].set_yticks([0, 0.0025, 0.005])
    axes[2].yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
    axes[2].set_xlim(-0.04, 2.15)
    axes[2].set_xticks([0, 0.5, 1, 1.5, 2])
    axes[2].set_xlabel("本轮追加训练决策数（百万步；0 = 已训练过的起点模型）", labelpad=10)
    fig.suptitle("继续训练后，成绩收敛了吗？", x=0.12, ha="left", y=0.98, fontsize=22)
    fig.text(
        0.12, 0.936, "KL 保留 · 三支模型 · 各追加 209.7 万步 · 原始验证点，不平滑", fontsize=13
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.105, 0.913),
        ncol=3,
        frameon=False,
        fontsize=12.2,
        handlelength=2.3,
        columnspacing=1.2,
    )
    fig.text(
        0.12,
        0.049,
        "纵轴已放大；折线只连接实测点。所有检查点使用相同验证局，存在有限样本误差。",
        fontsize=11.5,
    )
    fig.text(
        0.12,
        0.026,
        "这是验证曲线，不是正式测试的 80% 验收；强制局率低也不等于没有冻结／换位循环。",
        fontsize=11.5,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"validation-curves.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--font", type=Path, default=Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    )
    args = parser.parse_args()
    data = extract(args.run_directory.resolve())
    args.output_directory.mkdir(parents=True, exist_ok=False)
    draw(data, args.output_directory, args.font)
    (args.output_directory / "curve-evidence.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    rows = sum(
        m["episodes"]
        for c in data["curves"].values()
        for p in c["points"]
        for m in p["family_counts"].values()
    )
    print(
        json.dumps(
            {
                "checked_episode_rows": rows,
                "image": str((args.output_directory / "validation-curves.png").resolve()),
                "curves": {
                    name: {
                        "selected_steps": c["selected_steps"],
                        "hard_success": [p["hard_success"] for p in c["points"]],
                    }
                    for name, c in data["curves"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
