# Temporary script to run the environment with random actions and visualize the state

from environment.env import Env
import config
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os


PLOT_COLORS = {
    "ue": "#c9dfe2",
    "uav": "#e18283",
    "coverage": "#f6ad98",
    "association": "#bcd1c4",
    "collaboration": "#bdb6e4",
}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def plot_snapshot(env: Env, progress_step: int, save_dir: str) -> None:
    """Generates and saves a plot of the current environment state."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal")
    ax.set_title(f"Simulation Snapshot at Step: {progress_step}")
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")

    # Plot UEs
    ue_positions: np.ndarray = np.array([ue.pos for ue in env.ues])
    ax.scatter(ue_positions[:, 0], ue_positions[:, 1], c=PLOT_COLORS["ue"], marker=".", label="UEs")

    # Plot UAVs and their connections
    for uav in env.uavs:
        # UAV position
        ax.scatter(uav.pos[0], uav.pos[1], c=PLOT_COLORS["uav"], marker="s", s=100, label="UAV" if uav.id == 0 else "")

        # UAV coverage radius
        coverage_circle: Circle = Circle((uav.pos[0], uav.pos[1]), config.UAV_COVERAGE_RADIUS, color=PLOT_COLORS["coverage"], alpha=0.1)
        ax.add_patch(coverage_circle)

        # Lines to covered UEs
        for ue in uav.current_covered_ues:
            ax.plot(
                [uav.pos[0], ue.pos[0]],
                [uav.pos[1], ue.pos[1]],
                color=PLOT_COLORS["association"],
                linestyle="-",
                lw=0.5,
                label="UE Association" if "UE Association" not in plt.gca().get_legend_handles_labels()[1] else "",
            )

        # Line to collaborator (dashed magenta)
        if uav.current_collaborator:
            ax.plot(
                [uav.pos[0], uav.current_collaborator.pos[0]],
                [uav.pos[1], uav.current_collaborator.pos[1]],
                color=PLOT_COLORS["collaboration"],
                linestyle="--",
                lw=1.0,
                label="UAV Collaboration",
            )

    # Create a clean legend
    handles, labels = ax.get_legend_handles_labels()
    by_label: dict = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.savefig(f"{save_dir}/step_{progress_step:04d}.png")
    plt.close(fig)


def generate_random_actions(num_uavs: int) -> np.ndarray:
    """
    Generates random action vectors in [-1, 1] range for each UAV.
    
    Note: Collision avoidance is now handled by env._apply_actions_to_env(),
    so we only need to generate simple random actions here.
    """
    return np.random.uniform(-1, 1, (num_uavs, config.ACTION_DIM))


def main() -> None:
    env: Env = Env()
    vis_dir: str = "simulation_frames"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"Created directory: {vis_dir}")
    print("Starting simulation with random actions...")
    for t in range(config.STEPS_PER_EPISODE):
        actions: np.ndarray = generate_random_actions(config.NUM_UAVS)
        env.step(actions)
        if t % 50 == 0:
            plot_snapshot(env, t, vis_dir)
            print(f"Saved frame for time step {t}")
    print(f"\nSimulation finished. Visualization frames are saved in the '{vis_dir}' directory.")


if __name__ == "__main__":
    main()
