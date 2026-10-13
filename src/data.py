"""Data persistence for simulation results."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Config
from .simulation import SimulationResult, config_to_dict


REQUIRED_COLUMNS = [
    "step", "time", "x", "y", "u_x", "u_y",
    "fluid_u_x", "fluid_u_y", "rel_u_x", "rel_u_y",
    "drag_x", "drag_y", "buoyancy_x", "buoyancy_y",
    "acc_x", "acc_y",
    "dist_x_min", "dist_x_max", "dist_y_min", "dist_y_max"
]


def get_output_paths(config: Config, method: str) -> tuple[Path, Path]:
    """Get CSV and metadata file paths for a method."""
    output_dir = config.simulation.get_run_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{method}_results.csv"
    meta_path = output_dir / f"{method}_metadata.json"
    return csv_path, meta_path


def save_results(result: SimulationResult, config: Config) -> tuple[Path, Path]:
    """Save simulation results to CSV and metadata JSON."""
    csv_path, meta_path = get_output_paths(config, result.method)
    
    # Save CSV
    df = results_to_dataframe(result, config)
    df.to_csv(csv_path, index=False)
    
    # Save metadata
    metadata = {
        "method": result.method,
        "steps": result.steps,
        "termination_reason": result.termination_reason,
        "exit_time": result.exit_time,
        "exit_position": result.exit_position,
        "exit_velocity": result.exit_velocity,
        "config": result.config_snapshot,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return csv_path, meta_path


def load_results(method: str, config: Config) -> tuple[pd.DataFrame, dict]:
    """Load simulation results from CSV and metadata JSON."""
    csv_path, meta_path = get_output_paths(config, method)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
    
    df = pd.read_csv(csv_path)
    validate_dataframe(df)
    
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    
    return df, metadata


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that DataFrame has all required columns."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def results_to_dataframe(result: SimulationResult, config: Config) -> pd.DataFrame:
    """Convert simulation results to a pandas DataFrame with all diagnostics."""
    from .physics import compute_diagnostics
    
    records = []
    
    for i, (state, t) in enumerate(zip(result.states, result.times)):
        x, y, u_x, u_y = state
        diag = compute_diagnostics(state, config)
        
        record = {
            "step": i,
            "time": t,
            "x": x,
            "y": y,
            "u_x": u_x,
            "u_y": u_y,
            "fluid_u_x": diag["fluid_u_x"],
            "fluid_u_y": diag["fluid_u_y"],
            "rel_u_x": diag["rel_u_x"],
            "rel_u_y": diag["rel_u_y"],
            "drag_x": diag["drag_x"],
            "drag_y": diag["drag_y"],
            "buoyancy_x": diag["buoyancy_x"],
            "buoyancy_y": diag["buoyancy_y"],
            "acc_x": diag["acc_x"],
            "acc_y": diag["acc_y"],
            "dist_x_min": diag["dist_x_min"],
            "dist_x_max": diag["dist_x_max"],
            "dist_y_min": diag["dist_y_min"],
            "dist_y_max": diag["dist_y_max"],
        }
        records.append(record)
    
    return pd.DataFrame(records)


def save_summary(euler_meta: dict, rk4_meta: dict, config: Config) -> Path:
    """Save combined simulation summary."""
    output_dir = config.simulation.get_run_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "simulation_summary.json"
    
    summary = {
        "config": config_to_dict(config),
        "euler": euler_meta,
        "rk4": rk4_meta,
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary_path


def load_summary(config: Config) -> dict:
    """Load simulation summary."""
    summary_path = config.simulation.get_run_output_dir() / "simulation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    with open(summary_path, "r") as f:
        return json.load(f)


def check_results_exist(config: Config, methods: list[str]) -> dict[str, bool]:
    """Check which result files exist."""
    exists = {}
    for method in methods:
        csv_path, meta_path = get_output_paths(config, method)
        exists[method] = csv_path.exists() and meta_path.exists()
    return exists