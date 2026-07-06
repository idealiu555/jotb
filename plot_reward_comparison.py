"""
多算法训练 Reward 对比曲线图绘制工具
=====================================
读取多个算法的训练日志文件，绘制 reward 随训练进度的对比曲线图，
输出 PDF 和 SVG 两种格式。

用法示例：
python plot_reward_comparison.py \
  --files train_logs/amasac_log.json train_logs/masac_log.json train_logs/matd3_log.json train_logs/maddpg_log.json train_logs/mappo_log.json train_logs/random_log.json \
  --labels AMASAC MASAC MATD3 MADDPG MAPPO Random \
  --output_dir train_plots/comparison

"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ──────────────────────────────────────────────
# 学术论文风格配置（与 utils/plot_logs.py 保持一致）
# ──────────────────────────────────────────────
ACADEMIC_STYLE = {
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "axes.titleweight": "semibold",
    "axes.labelweight": "semibold",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.fontsize": 14,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

# 统一论文图调色板
PALETTE: tuple[str, ...] = (
    "#e18283",
    "#f6ad98",
    "#facd9d",
    "#bdb6e4",
    "#c9dfe2",
    "#bcd1c4",
)

AXIS_COLOR = "#3D4852"
TEXT_COLOR = "#263238"


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _paper_axes(ax) -> None:
    """对齐 plot_comparison.py 的坐标轴粗细和颜色。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=AXIS_COLOR, labelcolor=TEXT_COLOR)
    ax.set_axisbelow(True)


def smooth_curve(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """指数移动平均（EMA）平滑。"""
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def load_log(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    读取训练日志文件，返回 (x_array, reward_array)。

    支持：
    - JSON Lines 格式（每行一个 JSON 对象）
    - 单个 JSON 数组格式
    x 轴字段自动识别 ``episode`` 或 ``update``（兼容 MAPPO 历史日志）。
    """
    records: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as fh:
        first_char = fh.read(1)
        fh.seek(0)
        if first_char == "[":
            records = json.load(fh)
        else:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    if not records:
        raise ValueError(f"No valid data found in {file_path}")

    # 自动检测 x 轴字段
    first = records[0]
    if "episode" in first:
        x_key = "episode"
    elif "update" in first:
        x_key = "update"
    else:
        raise KeyError(f"Log file {file_path} has no 'episode' or 'update' field")

    x = np.array([r[x_key] for r in records], dtype=float)
    reward = np.array([r["reward"] for r in records], dtype=float)
    return x, reward


def _plot_algorithm(
    ax,
    x: np.ndarray,
    reward: np.ndarray,
    color: str,
    label: str,
    smoothing: float,
) -> None:
    """在 ax 上绘制单条算法曲线（原始细线 + EMA 粗线）。"""
    # 原始值（半透明细线）
    if len(reward) > 1:
        ax.plot(x, reward, color=color, linewidth=0.8, alpha=0.20, label="_nolegend_")
        # EMA 平滑线
        smoothed = smooth_curve(reward, smoothing)
        ax.plot(x, smoothed, color=color, linewidth=2.5, alpha=1.0, label=label)
    else:
        ax.plot(x, reward, color=color, linewidth=2.5, alpha=1.0, label=label)


# ──────────────────────────────────────────────
# 主绘图函数
# ──────────────────────────────────────────────

def plot_reward_comparison(
    log_files: list[str],
    labels: list[str],
    output_dir: str = "train_plots/comparison",
    output_name: str = "2_reward_comparison",
    smoothing: float = 0.9,
) -> None:
    """
    读取多个算法日志，绘制 reward 对比曲线并保存为 PDF 和 SVG。

    Args:
        log_files:   日志文件路径列表
        labels:      对应算法标签列表（与 log_files 等长）
        output_dir:  输出目录
        output_name: 输出文件名前缀（不含扩展名）
        smoothing:   EMA 平滑权重（0~1，越大越平滑）
    """
    if len(log_files) != len(labels):
        raise ValueError("log_files 与 labels 数量必须相同")
    if not log_files:
        raise ValueError("至少需要提供一个日志文件")

    os.makedirs(output_dir, exist_ok=True)

    # 加载各算法数据
    datasets: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for file_path, label in zip(log_files, labels):
        print(f"  Loading {label}: {file_path}")
        x, reward = load_log(file_path)
        color = PALETTE[len(datasets) % len(PALETTE)]
        datasets.append((x, reward, label, color))

    # ── 绘图 ──
    with plt.style.context("seaborn-v0_8-whitegrid"):
        plt.rcParams.update(ACADEMIC_STYLE)

        fig, ax = plt.subplots()

        for x, reward, label, color in datasets:
            _plot_algorithm(ax, x, reward, color, label, smoothing)

        ax.set_xlabel("Episode", labelpad=8, color=TEXT_COLOR)
        ax.set_ylabel("Average Team Reward", labelpad=8, color=TEXT_COLOR)
        ax.set_title("Training Reward Comparison", pad=11, color=TEXT_COLOR)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _paper_axes(ax)
        legend = ax.legend(
            loc="best",
            fancybox=True,
            shadow=False,
            handlelength=2.0,
        )
        for legend_line in legend.get_lines():
            legend_line.set_linewidth(5.0)

        fig.tight_layout()

        for fmt in ("pdf", "svg"):
            out_path = os.path.join(output_dir, f"{output_name}.{fmt}")
            fig.savefig(out_path, format=fmt, dpi=300, facecolor="white", edgecolor="none")
            print(f"  Saved: {out_path}")

        plt.close(fig)

    print("Done.")


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制多算法训练 Reward 对比曲线图，输出 PDF 和 SVG 格式。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        metavar="LOG_FILE",
        help="训练日志文件路径列表（JSON Lines 或 JSON 数组格式）",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        metavar="LABEL",
        help="对应算法标签列表，数量须与 --files 一致",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train_plots/comparison",
        help="输出目录（默认：train_plots/comparison）",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="reward_comparison",
        help="输出文件名前缀，不含扩展名（默认：reward_comparison）",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.9,
        help="EMA 平滑权重，范围 0~1，越大越平滑（默认：0.9）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_reward_comparison(
        log_files=args.files,
        labels=args.labels,
        output_dir=args.output_dir,
        output_name=args.output_name,
        smoothing=args.smoothing,
    )
