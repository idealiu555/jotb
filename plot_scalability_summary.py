"""Draw two compact scalability figures from UE-number test summaries.
python plot_scalability_summary.py
The script reads files named ``{ue}_{algorithm}_test_summary.json`` from
``test_logs`` and writes publication-ready latency and coverage figures to
``test_plots/scalability``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SUMMARY_RE = re.compile(r"(?P<ue>\d+)_(?P<algorithm>.+)_test_summary\.json$")

ALGORITHM_ORDER = ("amasac", "masac", "matd3", "maddpg", "mappo", "random")
DISPLAY_NAME = {
    "amasac": "AMASAC",
    "masac": "MASAC",
    "matd3": "MATD3",
    "maddpg": "MADDPG",
    "mappo": "MAPPO",
    "random": "Random",
}

# Keep the exact color sequence used by plot_comparison.py.
COMPARISON_COLORS: tuple[str, ...] = (
    "#e18283",
    "#f6ad98",
    "#facd9d",
    "#bdb6e4",
    "#c9dfe2",
    "#bcd1c4",
)

PALETTE = {
    algorithm: COMPARISON_COLORS[idx]
    for idx, algorithm in enumerate(ALGORITHM_ORDER)
}

MARKERS = {
    "amasac": "o",
    "masac": "v",
    "matd3": "s",
    "maddpg": "^",
    "mappo": "D",
    "random": "P",
}

TEXT = "#263238"
AXIS = "#3D4852"
GRID = "#E3E8EF"

CONFERENCE_STYLE: dict[str, object] = {
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "axes.titleweight": "semibold",
    "axes.labelweight": "semibold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "figure.edgecolor": "none",
    "axes.facecolor": "white",
    "axes.edgecolor": "none",
    "savefig.edgecolor": "none",
}

EXPORT_KWARGS: dict[str, object] = {
    "bbox_inches": "tight",
    "pad_inches": 0.04,
    "dpi": 300,
    "facecolor": "white",
    "edgecolor": "none",
}


def load_summary_data(input_dir: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(input_dir.glob("*_test_summary.json")):
        match = SUMMARY_RE.fullmatch(path.name)
        if not match:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        averages = payload.get("averages", {})
        missing = {"latency", "energy", "fairness"} - set(averages)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing averages: {missing_text}")
        rows.append(
            {
                "ue": int(match.group("ue")),
                "algorithm": match.group("algorithm"),
                "latency": float(averages["latency"]),
                "energy": float(averages["energy"]),
                "coverage": float(averages["fairness"]),
            }
        )
    if not rows:
        raise ValueError(f"No *_test_summary.json files found in {input_dir}")
    return rows


def style_matplotlib() -> None:
    mpl.rcParams.update(CONFERENCE_STYLE)
    mpl.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def save_pub_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {},
        ".tiff": {"dpi": 600},
    }.items():
        out = output_prefix.with_suffix(suffix)
        export_kwargs = {**EXPORT_KWARGS, **kwargs}
        fig.savefig(out, **export_kwargs)
        print(f"saved: {out}")


def draw_metric_figure(
    rows: list[dict[str, float | int | str]],
    metric: str,
    ylabel: str,
    title: str,
    output_prefix: Path,
) -> None:
    style_matplotlib()
    ues = sorted({int(row["ue"]) for row in rows})
    algorithms = [algo for algo in ALGORITHM_ORDER if any(row["algorithm"] == algo for row in rows)]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID, linewidth=0.75, alpha=0.85)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=AXIS, labelcolor=TEXT)

    for algorithm in algorithms:
        series = sorted(
            (row for row in rows if row["algorithm"] == algorithm),
            key=lambda row: int(row["ue"]),
        )
        x = [int(row["ue"]) for row in series]
        y = [float(row[metric]) for row in series]
        color = PALETTE[algorithm]
        is_focus = algorithm == "amasac"
        ax.plot(
            x,
            y,
            color=color,
            marker=MARKERS.get(algorithm, "o"),
            markersize=11.5 if is_focus else 11.0,
            markeredgecolor="white",
            markeredgewidth=1.45,
            linewidth=4.0 if is_focus else 3.2,
            alpha=0.98 if is_focus else 0.78,
            label=DISPLAY_NAME.get(algorithm, algorithm.upper()),
            zorder=4 if is_focus else 2,
        )

    values = [float(row[metric]) for row in rows]
    margin = (max(values) - min(values)) * 0.08
    ax.set_xlim(min(ues) - 8, max(ues) + 8)
    ax.set_ylim(min(values) - margin, max(values) + margin)
    ax.set_xticks(ues)
    ax.set_xlabel("Number of UEs", color=TEXT, labelpad=5)
    ax.set_ylabel(ylabel, color=TEXT, labelpad=5)
    ax.set_title(title, color=TEXT, pad=5)

    if metric == "coverage":
        ax.set_ylim(max(0.5, min(values) - margin), min(1.0, max(values) + margin))

    legend_handles = []
    for algorithm in algorithms:
        is_focus = algorithm == "amasac"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=PALETTE[algorithm],
                marker=MARKERS.get(algorithm, "o"),
                markersize=9.8 if is_focus else 9.4,
                markeredgecolor="white",
                markeredgewidth=1.25,
                linewidth=5.0,
                alpha=0.98 if is_focus else 0.78,
                label=DISPLAY_NAME.get(algorithm, algorithm.upper()),
            )
        )

    legend = ax.legend(
        handles=legend_handles,
        loc="best",
        ncol=2,
        columnspacing=1.0,
        handlelength=1.8,
        fontsize=14,
    )
    for line in legend.get_lines():
        line.set_linewidth(5.0)

    save_pub_figure(fig, output_prefix)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a one-panel UE scalability summary.")
    parser.add_argument("--input-dir", type=Path, default=Path("test_logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("test_plots/scalability"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_summary_data(args.input_dir)
    draw_metric_figure(
        rows,
        metric="latency",
        ylabel="Latency",
        title="Latency scalability",
        output_prefix=args.output_dir / "latency_vs_ue",
    )
    draw_metric_figure(
        rows,
        metric="coverage",
        ylabel="Coverage",
        title="Coverage scalability",
        output_prefix=args.output_dir / "coverage_vs_ue",
    )
    draw_metric_figure(
        rows,
        metric="energy",
        ylabel="Energy",
        title="Energy scalability",
        output_prefix=args.output_dir / "energy_vs_ue",
    )


if __name__ == "__main__":
    main()
