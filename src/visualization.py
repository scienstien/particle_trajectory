"""Static visualization for Lagrangian Particle Tracking."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config


def plot_fluid_field(
    ax: plt.Axes,
    config: Config,
    grid_resolution: int = 15,
    scale: float = 0.3
) -> None:
    """Plot the fluid velocity field using quiver."""
    x = np.linspace(config.domain.x_min, config.domain.x_max, grid_resolution)
    y = np.linspace(config.domain.y_min, config.domain.y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Fluid velocity: u = 1, v = sin(x^2 + y^2)
    U = np.ones_like(X)
    V = np.sin(X**2 + Y**2)
    
    # Use smaller, semi-transparent arrows to avoid obscuring trajectory
    ax.quiver(X, Y, U, V, scale=scale, alpha=0.3, color="blue", width=0.004, 
              headwidth=3, headlength=4, headaxislength=3)


def plot_trajectory(
    ax: plt.Axes,
    df: pd.DataFrame,
    method: str,
    color: str = "red",
    linewidth: float = 1.5,
    linestyle: str = "-",
    label: Optional[str] = None
) -> None:
    """Plot particle trajectory."""
    if label is None:
        label = method.upper()
    ax.plot(df["x"], df["y"], color=color, linewidth=linewidth, linestyle=linestyle, label=label)


def plot_initial_exit(
    ax: plt.Axes,
    df: pd.DataFrame,
    metadata: dict,
    method: str
) -> None:
    """Mark initial and exit positions."""
    # Initial position
    ax.plot(df["x"].iloc[0], df["y"].iloc[0], "go", markersize=10, label="Start")
    
    # Exit position
    exit_pos = metadata.get("exit_position")
    if exit_pos:
        ax.plot(exit_pos[0], exit_pos[1], "ro", markersize=10, label="Exit")


def create_flow_and_trajectory_plot(
    df: pd.DataFrame,
    metadata: dict,
    config: Config,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create the required assignment plot: fluid field with particle trajectory.
    
    This is the main plot showing the fluid velocity field with the
    particle trajectory as a red line.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Fluid field
    plot_fluid_field(ax, config)
    
    # Trajectory (red line as required) - thicker line for visibility
    plot_trajectory(ax, df, metadata["method"], color="red", linewidth=2.5, label="Trajectory")
    
    # Initial and exit positions
    plot_initial_exit(ax, df, metadata, metadata["method"])
    
    # Formatting
    ax.set_xlim(config.domain.x_min, config.domain.x_max)
    ax.set_ylim(config.domain.y_min, config.domain.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Particle Trajectory in Fluid Field ({metadata['method'].upper()})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    
    return fig


def create_method_comparison_plot(
    euler_df: pd.DataFrame,
    euler_meta: dict,
    rk4_df: pd.DataFrame,
    rk4_meta: dict,
    config: Config,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Create comparison plot showing both Euler and RK4 trajectories."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Fluid field (lighter)
    plot_fluid_field(ax, config, scale=0.4)
    
    # Trajectories - distinct styles
    plot_trajectory(ax, euler_df, "euler", color="#FF00FF", linewidth=2, linestyle="--", label="Euler (dashed)")
    plot_trajectory(ax, rk4_df, "rk4", color="red", linewidth=2.5, linestyle="-", label="RK4 (solid)")
    
    # Initial position (shared)
    ax.plot(euler_df["x"].iloc[0], euler_df["y"].iloc[0], "go", markersize=12, label="Start")
    
    # Exit positions
    for meta, color, label in [
        (euler_meta, "#FF00FF", "Euler Exit"),
        (rk4_meta, "red", "RK4 Exit")
    ]:
        exit_pos = meta.get("exit_position")
        if exit_pos:
            ax.plot(exit_pos[0], exit_pos[1], "o", color=color, markersize=10, label=label)
    
    # Formatting
    ax.set_xlim(config.domain.x_min, config.domain.x_max)
    ax.set_ylim(config.domain.y_min, config.domain.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Method Comparison: Euler vs RK4")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Add numerical comparison as text box
    euler_exit = euler_meta.get("exit_time", 0)
    rk4_exit = rk4_meta.get("exit_time", 0)
    if euler_meta.get("exit_position") and rk4_meta.get("exit_position"):
        ex, ey = euler_meta["exit_position"]
        rx, ry = rk4_meta["exit_position"]
        time_diff = abs(rk4_exit - euler_exit)
        pos_diff = ((rx - ex)**2 + (ry - ey)**2)**0.5
        
        info_text = (
            f"Euler:  t_exit = {euler_exit:.6f} s, pos = ({ex:.6f}, {ey:.6f}) m\n"
            f"RK4:    t_exit = {rk4_exit:.6f} s, pos = ({rx:.6f}, {ry:.6f}) m\n"
            f"Δt = {time_diff:.6f} s, Δpos = {pos_diff:.6f} m"
        )
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                verticalalignment="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                fontfamily="monospace")
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    
    return fig


def create_velocity_comparison_plot(
    euler_df: pd.DataFrame,
    rk4_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Create velocity comparison over time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # X velocity
    axes[0].plot(euler_df["time"], euler_df["u_x"], "r--", label="Euler", alpha=0.8)
    axes[0].plot(rk4_df["time"], rk4_df["u_x"], "b-", label="RK4", alpha=0.8)
    axes[0].set_ylabel("u_x (m/s)")
    axes[0].set_title("Particle Velocity Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Y velocity
    axes[1].plot(euler_df["time"], euler_df["u_y"], "r--", label="Euler", alpha=0.8)
    axes[1].plot(rk4_df["time"], rk4_df["u_y"], "b-", label="RK4", alpha=0.8)
    axes[1].set_ylabel("u_y (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    
    return fig


def create_numerical_comparison_plot(
    euler_df: pd.DataFrame,
    euler_meta: dict,
    rk4_df: pd.DataFrame,
    rk4_meta: dict,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Create numerical comparison table as a plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    
    euler_exit = euler_meta.get("exit_time", 0)
    rk4_exit = rk4_meta.get("exit_time", 0)
    
    table_data = [
        ["Metric", "Euler", "RK4", "Absolute Difference"],
        ["Exit Time (s)", f"{euler_exit:.6f}", f"{rk4_exit:.6f}", f"{abs(rk4_exit - euler_exit):.6f}"],
    ]
    
    if euler_meta.get("exit_position") and rk4_meta.get("exit_position"):
        ex, ey = euler_meta["exit_position"]
        rx, ry = rk4_meta["exit_position"]
        pos_diff = ((rx - ex)**2 + (ry - ey)**2)**0.5
        table_data.extend([
            ["Exit X (m)", f"{ex:.6f}", f"{rx:.6f}", f"{abs(rx - ex):.6f}"],
            ["Exit Y (m)", f"{ey:.6f}", f"{ry:.6f}", f"{abs(ry - ey):.6f}"],
            ["Exit Position Distance (m)", "", "", f"{pos_diff:.6f}"],
        ])
    
    if euler_meta.get("exit_velocity") and rk4_meta.get("exit_velocity"):
        evx, evy = euler_meta["exit_velocity"]
        rvx, rvy = rk4_meta["exit_velocity"]
        table_data.extend([
            ["Exit Ux (m/s)", f"{evx:.6f}", f"{rvx:.6f}", f"{abs(rvx - evx):.6f}"],
            ["Exit Uy (m/s)", f"{evy:.6f}", f"{rvy:.6f}", f"{abs(rvy - evy):.6f}"],
        ])
    
    table_data.extend([
        ["Steps", str(euler_meta.get("steps", 0)), str(rk4_meta.get("steps", 0)), ""],
        ["Termination", euler_meta.get("termination_reason", ""), rk4_meta.get("termination_reason", ""), ""],
    ])
    
    table = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white")
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#D6E4F0")
            else:
                cell.set_facecolor("white")
    
    ax.set_title("Numerical Comparison: Euler vs RK4", fontsize=14, pad=20)
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    
    return fig


def create_position_error_plot(
    euler_df: pd.DataFrame,
    rk4_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Create position error vs time plot (Euler relative to RK4 as reference)."""
    # Interpolate to common time grid
    min_len = min(len(euler_df), len(rk4_df))
    if min_len < 2:
        return None
    
    # Use common time steps (both use same dt)
    time = euler_df["time"].iloc[:min_len].values
    pos_error_x = np.abs(euler_df["x"].iloc[:min_len].values - rk4_df["x"].iloc[:min_len].values)
    pos_error_y = np.abs(euler_df["y"].iloc[:min_len].values - rk4_df["y"].iloc[:min_len].values)
    pos_error = np.sqrt(pos_error_x**2 + pos_error_y**2)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.semilogy(time, pos_error, "b-", linewidth=1.5, label="Position Error (Euler vs RK4)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Euler Position Error Relative to RK4 (Reference)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    
    return fig


def generate_all_plots(
    config: Config,
    euler_df: pd.DataFrame,
    euler_meta: dict,
    rk4_df: pd.DataFrame,
    rk4_meta: dict,
    output_dir: Optional[Path] = None
) -> list[Path]:
    """Generate all required plots and return list of output paths."""
    if output_dir is None:
        output_dir = config.simulation.get_run_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    
    # Main assignment plot (fluid field + trajectory)
    path1 = output_dir / "flow_and_trajectory.png"
    create_flow_and_trajectory_plot(euler_df, euler_meta, config, path1)
    plt.close()
    paths.append(path1)
    
    # Method comparison
    path2 = output_dir / "method_comparison.png"
    create_method_comparison_plot(euler_df, euler_meta, rk4_df, rk4_meta, config, path2)
    plt.close()
    paths.append(path2)
    
    # Numerical comparison table
    path3 = output_dir / "numerical_comparison.png"
    create_numerical_comparison_plot(euler_df, euler_meta, rk4_df, rk4_meta, path3)
    plt.close()
    paths.append(path3)
    
    # Position error vs time
    path4 = output_dir / "position_error.png"
    err_plot = create_position_error_plot(euler_df, rk4_df, path4)
    if err_plot is not None:
        plt.close()
        paths.append(path4)
    
    # Velocity comparison
    path5 = output_dir / "velocity_comparison.png"
    create_velocity_comparison_plot(euler_df, rk4_df, path5)
    plt.close()
    paths.append(path5)
    
    # Individual method plots
    for df, meta in [(euler_df, euler_meta), (rk4_df, rk4_meta)]:
        path = output_dir / f"{meta['method']}_trajectory.png"
        create_flow_and_trajectory_plot(df, meta, config, path)
        plt.close()
        paths.append(path)
    
    return paths