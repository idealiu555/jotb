"""
python plot_comparison.py \
  --files model_test/comparison_data/summary_amasac.json \
          model_test/comparison_data/summary_masac.json \
          model_test/comparison_data/summary_matd3.json \
          model_test/comparison_data/summary_maddpg.json \
          model_test/comparison_data/summary_mappo.json \
          model_test/comparison_data/summary_random.json \
  --labels AMASAC MASAC MATD3 MADDPG MAPPO Random

   默认输出到 model_test/comparison_plots 目录下，自动生成命名为算法标签和指标名称的 SVG / PDF 图片。
"""
import argparse
import json
import os

import numpy as np


CONFERENCE_STYLE: dict[str, object] = {
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "axes.titleweight": "semibold",
    "axes.labelweight": "semibold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
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

METRIC_CONFIG: dict[str, dict[str, str]] = {
    "reward": {"ylabel": "Average Reward", "title": "Reward"},
    "latency": {"ylabel": "Average Latency (s)", "title": "Latency"},
    "energy": {"ylabel": "Average Energy Consumption (J)", "title": "Energy"},
    "fairness": {"ylabel": "Jain's Fairness Index", "title": "Fairness"},
    "rate": {"ylabel": "Average System Throughput (bps)", "title": "Throughput"},
    "collisions": {"ylabel": "Average Collision Count", "title": "Collisions"},
    "boundaries": {"ylabel": "Average Boundary Violation Count", "title": "Boundaries"},
}

COLORS: tuple[str, ...] = (
    "#e18283",
    "#f6ad98",
    "#facd9d",
    "#bdb6e4",
    "#c9dfe2",
    "#bcd1c4",
)

AXIS_COLOR = "#3D4852"
GRID_COLOR = "#E3E8EF"
TEXT_COLOR = "#263238"


def _load_summary_averages(file_path: str) -> dict[str, float]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    averages = data.get("averages")
    if not isinstance(averages, dict):
        raise ValueError(f"{file_path} does not contain an 'averages' object")

    result: dict[str, float] = {}
    for metric in METRIC_CONFIG:
        if metric not in averages:
            raise ValueError(f"{file_path} is missing averages.{metric}")
        result[metric] = float(averages[metric])
    return result




def _format_bar_label(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e6 or abs(value) < 1e-3:
        return f"{value:.2e}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _paper_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=AXIS_COLOR, labelcolor=TEXT_COLOR)
    ax.set_axisbelow(True)


def _save_figure(fig, output_path: str) -> None:
    path_no_ext = os.path.splitext(output_path)[0]
    for fmt in ("svg", "pdf"):
        out = f"{path_no_ext}.{fmt}"
        fig.savefig(out, format=fmt, **EXPORT_KWARGS)
        print(f"saved: {out}")


def _remove_outer_frame(fig, ax) -> None:
    fig.patch.set_edgecolor("none")
    fig.patch.set_linewidth(0)
    fig.patch.set_antialiased(False)
    ax.patch.set_visible(False)
    ax.patch.set_edgecolor("none")
    ax.patch.set_linewidth(0)
    ax.patch.set_antialiased(False)


def _disable_clipping(ax) -> None:
    for artist in ax.get_children():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)


def _draw_horizontal_grid(ax) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for tick in ax.get_yticks():
        if tick < ymin or tick > ymax:
            continue
        ax.hlines(
            tick,
            xmin,
            xmax,
            colors=GRID_COLOR,
            linewidth=0.75,
            alpha=0.85,
            zorder=0,
            clip_on=False,
        )


def _draw_full_grid(ax) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for tick in ax.get_xticks():
        if tick < xmin or tick > xmax:
            continue
        ax.vlines(
            tick,
            ymin,
            ymax,
            colors=GRID_COLOR,
            linewidth=0.75,
            alpha=0.85,
            zorder=0,
            clip_on=False,
        )
    for tick in ax.get_yticks():
        if tick < ymin or tick > ymax:
            continue
        ax.hlines(
            tick,
            xmin,
            xmax,
            colors=GRID_COLOR,
            linewidth=0.75,
            alpha=0.85,
            zorder=0,
            clip_on=False,
        )


def _value_margin(values: list[float], ratio: float, minimum: float) -> float:
    data_range = max(values) - min(values)
    return data_range * ratio + minimum


def plot_metric_bar(
    labels: list[str],
    values: list[float],
    metric: str,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    cfg = METRIC_CONFIG[metric]
    x = np.arange(len(labels))
    width = 0.56 if len(labels) <= 3 else 0.64
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]

    fig_width = max(6.2, 1.25 * len(labels) + 2.4)
    with plt.rc_context(CONFERENCE_STYLE):
        fig, ax = plt.subplots(figsize=(fig_width, 4.8))
        _remove_outer_frame(fig, ax)
        bars = ax.bar(
            x,
            values,
            width=width,
            color=colors,
            edgecolor="white",
            linewidth=1.1,
            alpha=0.92,
        )

        ax.set_title(f"{cfg['title']} Comparison", pad=11, color=TEXT_COLOR)
        ax.set_ylabel(cfg["ylabel"], labelpad=8, color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        _paper_axes(ax)

        max_value = max(values) if values else 0.0
        min_value = min(values) if values else 0.0
        if min_value >= 0:
            ax.set_ylim(bottom=0)
        if values:
            ax.margins(y=0.18)
        _draw_horizontal_grid(ax)

        label_color = "#30363D"
        for bar, value in zip(bars, values):
            height = bar.get_height()
            offset = 5 if height >= 0 else -14
            va = "bottom" if height >= 0 else "top"
            ax.annotate(
                _format_bar_label(value),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=11,
                color=label_color,
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.tight_layout()
        _disable_clipping(ax)
        _save_figure(fig, output_path)
        plt.close(fig)


def _label_index(labels: list[str], target: str) -> int | None:
    target_lower = target.strip().lower()
    for idx, label in enumerate(labels):
        if label.strip().lower() == target_lower:
            return idx
    return None


def plot_energy_latency_bubble(
    labels: list[str],
    summaries: list[dict[str, float]],
    output_dir: str = "model_test/comparison_plots",
) -> None:
    """绘制能耗-时延气泡图，气泡大小表示 JFI。"""
    import matplotlib.pyplot as plt

    energy_vals = [summary["energy"] for summary in summaries]
    latency_vals = [summary["latency"] for summary in summaries]
    fairness_vals = [summary["fairness"] for summary in summaries]
    random_idx = _label_index(labels, "random")
    axis_indices = [
        idx for idx in range(len(labels))
        if random_idx is None or idx != random_idx
    ]
    if not axis_indices:
        axis_indices = list(range(len(labels)))
    axis_energy_vals = [energy_vals[idx] for idx in axis_indices]
    axis_latency_vals = [latency_vals[idx] for idx in axis_indices]

    # Map JFI to visual tiers instead of raw differences. The JFI values are often
    # numerically close, so rank-based sizing makes fairness levels readable.
    min_bubble_size = 420.0
    max_bubble_size = 2300.0
    if len(fairness_vals) > 1:
        sorted_indices = sorted(range(len(fairness_vals)), key=lambda idx: fairness_vals[idx])
        size_step = (max_bubble_size - min_bubble_size) / (len(fairness_vals) - 1)
        bubble_sizes = [min_bubble_size] * len(fairness_vals)
        for rank, idx in enumerate(sorted_indices):
            bubble_sizes[idx] = min_bubble_size + rank * size_step
    else:
        bubble_sizes = [0.5 * (min_bubble_size + max_bubble_size)]

    with plt.rc_context(CONFERENCE_STYLE):
        fig, ax = plt.subplots(figsize=(7.3, 5.3))
        _remove_outer_frame(fig, ax)

        label_offsets = {
            "amasac": (28, 0, "left", "center"),
            "masac": (28, 0, "left", "center"),
            "matd3": (28, 0, "left", "center"),
            "mappo": (28, 0, "left", "center"),
            "random": (28, 0, "left", "center"),
        }

        energy_margin = _value_margin(axis_energy_vals, ratio=0.18, minimum=5.0)
        lat_margin = _value_margin(axis_latency_vals, ratio=0.16, minimum=80.0)
        x_min = min(axis_energy_vals) - energy_margin
        x_max = max(axis_energy_vals) + energy_margin
        y_min = min(axis_latency_vals) - lat_margin
        y_max = max(axis_latency_vals) + lat_margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        x_span = x_max - x_min
        y_span = y_max - y_min

        for i, (label, energy, latency, size) in enumerate(
            zip(labels, energy_vals, latency_vals, bubble_sizes)
        ):
            color = COLORS[i % len(COLORS)]
            is_outlier = (
                random_idx is not None
                and i == random_idx
                and (energy < x_min or energy > x_max or latency < y_min or latency > y_max)
            )
            plot_energy = energy
            plot_latency = latency
            plot_label = label
            if is_outlier:
                plot_energy = x_max - 0.075 * x_span
                plot_latency = y_max - 0.090 * y_span
                plot_label = f"{label}\n(outlier)"
                ax.annotate(
                    "",
                    xy=(x_max - 0.008 * x_span, y_max - 0.008 * y_span),
                    xytext=(plot_energy, plot_latency),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#68777D",
                        "linewidth": 1.2,
                        "mutation_scale": 11,
                    },
                    zorder=2,
                )
            ax.scatter(
                plot_energy,
                plot_latency,
                s=size,
                color=color,
                alpha=0.94,
                edgecolors="white",
                linewidths=1.8,
                zorder=3,
            )

            dx, dy, ha, va = label_offsets.get(label.strip().lower(), (12, 0, "left", "center"))
            ax.annotate(
                plot_label,
                xy=(plot_energy, plot_latency),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=11,
                fontweight="semibold",
                color=TEXT_COLOR,
                zorder=4,
            )

        ax.set_xlabel("Average Energy Consumption (J)", labelpad=8, color=TEXT_COLOR)
        ax.set_ylabel("Average Latency (s)", labelpad=8, color=TEXT_COLOR)
        ax.set_title("Energy-Latency-Fairness Trade-off", pad=11, color=TEXT_COLOR)

        _paper_axes(ax)
        _draw_full_grid(ax)

        arrow_head = (0.055, 0.065)
        arrow_tail = (0.190, 0.205)
        head_px = ax.transAxes.transform(arrow_head)
        tail_px = ax.transAxes.transform(arrow_tail)
        arrow_text_angle = float(
            np.degrees(np.arctan2(tail_px[1] - head_px[1], tail_px[0] - head_px[0]))
        )

        ax.annotate(
            "",
            xy=arrow_head,
            xytext=arrow_tail,
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#68777D",
                "linewidth": 1.2,
                "mutation_scale": 11,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=7,
        )
        ax.annotate(
            "Lower latency",
            xy=(0.062, 0.164),
            xycoords="axes fraction",
            ha="left",
            va="center",
            rotation=arrow_text_angle,
            rotation_mode="anchor",
            fontsize=8.6,
            color="#68777D",
            fontfamily=["Comic Sans MS", "Segoe Print", "cursive"],
            zorder=7,
        )
        ax.annotate(
            "Lower energy",
            xy=(0.124, 0.054),
            xycoords="axes fraction",
            ha="left",
            va="center",
            rotation=arrow_text_angle,
            rotation_mode="anchor",
            fontsize=8.6,
            color="#68777D",
            fontfamily=["Comic Sans MS", "Segoe Print", "cursive"],
            zorder=7,
        )

        os.makedirs(output_dir, exist_ok=True)
        fig.tight_layout()
        _disable_clipping(ax)
        base_path = os.path.join(output_dir, "energy_latency_fairness")
        _save_figure(fig, base_path)
        plt.close(fig)


def plot_algorithm_comparison(
    summary_files: list[str],
    labels: list[str],
    output_dir: str = "model_test/comparison_plots",
) -> None:
    if len(summary_files) != len(labels):
        raise ValueError("The number of files must match the number of labels")
    if not summary_files:
        raise ValueError("At least one summary file is required")

    summaries = [_load_summary_averages(file_path) for file_path in summary_files]

    for metric in METRIC_CONFIG:
        values = [summary[metric] for summary in summaries]
        output_path = os.path.join(output_dir, f"{metric}_comparison")
        plot_metric_bar(labels, values, metric, output_path)

    # 气泡图：能耗 vs 时延，气泡大小表示 JFI
    plot_energy_latency_bubble(labels, summaries, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制多算法测试 summary 指标柱形对比图。")
    parser.add_argument("--files", nargs="+", required=True, help="测试 summary JSON 文件路径列表")
    parser.add_argument("--labels", nargs="+", required=True, help="对应算法标签，数量必须与文件一致")
    parser.add_argument("--output_dir", type=str, default="model_test/comparison_plots", help="SVG 输出目录")

    args = parser.parse_args()
    plot_algorithm_comparison(args.files, args.labels, args.output_dir)
