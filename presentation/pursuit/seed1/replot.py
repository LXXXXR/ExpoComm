"""Replot the bundled seed 1 CSVs. Requires matplotlib; no repository or network access."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
TEAM_SIZES = (25, 45, 61)


def read_csv(name):
    with (HERE / "data" / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(int(r["training_seed"]) != 1 for r in rows):
        raise ValueError(f"Expected only training seed 1 in {name}")
    return rows


def save(fig, name, output, formats):
    for suffix in formats:
        fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def return_curves(rows, color):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, n in zip(axes, TEAM_SIZES):
        run = sorted((r for r in rows if int(r["agents"]) == n), key=lambda r: int(r["timestep"]))
        if [int(r["timestep"]) for r in run] != list(range(200, 5000201, 50000)):
            raise ValueError(f"Incomplete or duplicate evaluation trajectory for {n} agents")
        x = [int(r["timestep"]) / 1e6 for r in run]
        y = [float(r["test_return_mean"]) for r in run]
        sd = [float(r["test_return_std_over_100_episodes"]) for r in run]
        ax.fill_between(x, [m - s for m, s in zip(y, sd)],
                        [m + s for m, s in zip(y, sd)], color=color, alpha=0.16)
        ax.plot(x, y, color=color, linewidth=2)
        ax.scatter(x[-1], y[-1], color=color, s=30, zorder=3)
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(-2, 15),
                    textcoords="offset points", ha="right", color=color, weight="bold")
        size = int(run[0]["map_size"])
        ax.set_title(f"{n} agents · {size} × {size} map")
        ax.set(xlim=(-0.08, 5.1), ylim=(-45, 100), xticks=range(6), yticks=[-40, 0, 40, 80])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test_return_mean")
    fig.suptitle("Adversarial Pursuit · Static ExpoComm · Training seed 1", fontsize=17)
    fig.supxlabel("Training environment steps (millions)", y=0.09)
    fig.text(0.5, 0.025, "Unsmoothed means over 100 episodes; shading: ±1 episode SD. Final checkpoint: 5,000,200.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    return fig


def behaviour_metrics(rows, color):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column, title in zip(
        axes,
        ("tag_action_pct", "rewarded_tag_success_pct"),
        ("Tag actions / decisions (%)", "Rewarded tags / attempts (%)"),
    ):
        for i, n in enumerate(TEAM_SIZES):
            selected = [r for r in rows if int(r["agents"]) == n]
            if len(selected) != 2 or {r["checkpoint"] for r in selected} != {"initial", "final"}:
                raise ValueError(f"Expected one initial and one final probe for {n} agents")
            pair = {r["checkpoint"]: r for r in selected}
            initial, final = (float(pair[k][column]) for k in ("initial", "final"))
            y = 2 - i
            ax.plot([initial, final], [y, y], color="0.65", linewidth=2)
            ax.scatter(initial, y, facecolors="none", edgecolors="0.55", s=65, linewidths=1.5, zorder=3)
            ax.scatter(final, y, color=color, s=65, zorder=3)
            dx = -4 if column == "tag_action_pct" else 4
            ax.text(final + dx, y, f"{final:.1f}", va="center",
                    ha="right" if dx < 0 else "left", color=color, weight="bold")
        ax.set(xlim=(-4, 114), ylim=(-0.5, 2.5), xticks=[0, 50, 100],
               yticks=[2, 1, 0], yticklabels=[f"{n} agents" for n in TEAM_SIZES])
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Pursuit behaviour · Training seed 1", fontsize=17)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", color="0.55", label="Initial (200 training steps)"),
        Line2D([], [], marker="o", linestyle="none", color=color, label="Final (5,000,200 training steps)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.09), ncol=2, frameon=False)
    fig.text(0.5, 0.025, "One matched probe episode per team size; rewarded tag events are inferred, not unique captures or wins.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.19, 1, 0.92))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", default="default", help="Matplotlib style name or path to a .mplstyle file")
    parser.add_argument("--color", default="#2A6EBB", help="Curve and final-checkpoint colour")
    parser.add_argument("--output", type=Path, default=HERE / "replotted", help="Output directory (default: replotted beside this script)")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf", "svg"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.style.context(args.style):
        plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
        save(return_curves(read_csv("seed1_evaluations.csv"), args.color), "test_return_mean_seed1", args.output, args.formats)
        save(behaviour_metrics(read_csv("seed1_behaviour_metrics.csv"), args.color), "behaviour_metrics_seed1", args.output, args.formats)
    print(f"Saved graphs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
