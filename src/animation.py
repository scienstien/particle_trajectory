"""Animation for Lagrangian Particle Tracking."""

from pathlib import Path
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .visualization import plot_fluid_field


def validate_animation_data(df: pd.DataFrame, metadata: dict, config: Config) -> bool:
    """
    Validate that animation data is consistent.
    
    Returns True if validation passes, False otherwise.
    """
    # Check that time is monotonically increasing
    times = df["time"].values
    if not np.all(np.diff(times) > 0):
        print("WARNING: Time values are not monotonically increasing")
        return False
    
    # Check that trajectory doesn't go outside domain (except possibly last point)
    # Note: The last point might be outside if boundary_exit
    termination = metadata.get("termination_reason", "")
    if termination == "boundary_exit":
        # Check all but last point are inside
        for i in range(len(df) - 1):
            x, y = df["x"].iloc[i], df["y"].iloc[i]
            if not (config.domain.x_min <= x <= config.domain.x_max and 
                    config.domain.y_min <= y <= config.domain.y_max):
                print(f"WARNING: Point {i} at ({x}, {y}) is outside domain")
                return False
    
    return True


def prepare_animation_data(df: pd.DataFrame, metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare trajectory data for animation.
    
    Uses the full trajectory from CSV, but replaces the final point with
    the interpolated exit position from metadata if available and different.
    
    Returns (x_data, y_data, times) arrays.
    """
    x_data = df["x"].values.copy()
    y_data = df["y"].values.copy()
    times = df["time"].values.copy()
    
    # If we have an interpolated exit position, use it for the final frame
    # This ensures the animation ends exactly at the boundary crossing
    exit_pos = metadata.get("exit_position")
    if exit_pos is not None:
        final_x = x_data[-1]
        final_y = y_data[-1]
        tol = 1e-6
        # If the CSV final position differs significantly from interpolated exit,
        # replace the last point with the interpolated position
        if abs(final_x - exit_pos[0]) > tol or abs(final_y - exit_pos[1]) > tol:
            x_data[-1] = exit_pos[0]
            y_data[-1] = exit_pos[1]
            # Keep the exit time from metadata if available
            if metadata.get("exit_time") is not None:
                times[-1] = metadata["exit_time"]
    
    return x_data, y_data, times


def create_animation(
    df: pd.DataFrame,
    metadata: dict,
    config: Config,
    output_path: Optional[Path] = None,
    interval: int = 20,
    fps: int = 30,
    dpi: int = 100,
    frame_skip: int = 1,
    validate: bool = True
) -> animation.FuncAnimation:
    """
    Create particle animation from simulation data.
    
    Args:
        df: Simulation DataFrame with trajectory data
        metadata: Simulation metadata
        config: Configuration
        output_path: Optional path to save animation
        interval: Milliseconds between frames
        fps: Frames per second for saved animation
        dpi: DPI for saved animation
        frame_skip: Skip frames (e.g., 5 means every 5th frame)
        validate: Whether to validate data before creating animation
    
    Returns:
        FuncAnimation object
    """
    if validate:
        if not validate_animation_data(df, metadata, config):
            raise ValueError("Animation data validation failed - see warnings above")
    
    # Prepare trajectory data (use interpolated exit position for final frame)
    x_data, y_data, times = prepare_animation_data(df, metadata)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot fluid field - use same settings as static plots
    plot_fluid_field(ax, config)
    
    # Initialize trajectory line
    trajectory_line, = ax.plot([], [], "r-", linewidth=2, label="Trajectory")
    
    # Initialize particle marker
    particle, = ax.plot([], [], "ro", markersize=12, label="Particle", zorder=5)
    
    # Initial position marker
    ax.plot(df["x"].iloc[0], df["y"].iloc[0], "go", markersize=12, label="Start", zorder=5)
    
    # Exit position marker (from metadata - interpolated)
    exit_pos = metadata.get("exit_position")
    if exit_pos:
        ax.plot(exit_pos[0], exit_pos[1], "k*", markersize=15, label="Exit", zorder=5)
    
    # Time text
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, 
                        verticalalignment="top", fontsize=12,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    # Formatting
    ax.set_xlim(config.domain.x_min, config.domain.x_max)
    ax.set_ylim(config.domain.y_min, config.domain.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Particle Animation ({metadata['method'].upper()})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Apply frame skip - ensure final frame is always included
    indices = np.arange(0, len(x_data), frame_skip)
    # Ensure the last frame is included
    if indices[-1] != len(x_data) - 1:
        indices = np.append(indices, len(x_data) - 1)
    x_data = x_data[indices]
    y_data = y_data[indices]
    times = times[indices]
    
    def init():
        trajectory_line.set_data([], [])
        particle.set_data([], [])
        time_text.set_text("")
        return trajectory_line, particle, time_text
    
    def animate(frame):
        # Draw trajectory up to current frame
        trajectory_line.set_data(x_data[:frame+1], y_data[:frame+1])
        
        # Update particle position
        particle.set_data([x_data[frame]], [y_data[frame]])
        
        # Update time text
        time_text.set_text(f"Time: {times[frame]:.3f} s")
        
        return trajectory_line, particle, time_text
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(x_data),
        interval=interval,
        blit=True,
        repeat=False
    )
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to save as GIF first (no external dependency)
        try:
            if output_path.suffix.lower() == ".gif":
                anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
            else:
                # Try ffmpeg for mp4
                anim.save(output_path, writer="ffmpeg", fps=fps, dpi=dpi)
            
            # Validate final frame after saving
            if validate and exit_pos is not None:
                final_x = x_data[-1]
                final_y = y_data[-1]
                tol = 1e-6
                if abs(final_x - exit_pos[0]) > tol or abs(final_y - exit_pos[1]) > tol:
                    print(f"WARNING: Animation final frame ({final_x:.6f}, {final_y:.6f}) "
                          f"differs from exit position ({exit_pos[0]:.6f}, {exit_pos[1]:.6f})")
                else:
                    print(f"Animation validation passed: final frame matches exit position "
                          f"({final_x:.6f}, {final_y:.6f})")
        except Exception as e:
            print(f"Warning: Could not save animation: {e}")
            print("Install pillow for GIF or ffmpeg for MP4.")
    
    return anim


def animate_from_files(
    method: str,
    config: Config,
    output_path: Optional[Path] = None,
    frame_skip: int = 1,
    **kwargs
) -> animation.FuncAnimation:
    """
    Load simulation data and create animation.
    
    This function loads existing CSV data and creates animation
    without running the simulation.
    """
    from .data import load_results
    
    df, metadata = load_results(method, config)
    return create_animation(df, metadata, config, output_path, frame_skip=frame_skip, **kwargs)