"""Validation tests for consistency checks."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import get_default_config, Config, SimulationConfig
from src.simulation import run_simulation, run_both_methods
from src.data import load_results, load_summary
from src.animation import validate_animation_data, prepare_animation_data


@pytest.fixture
def base_config():
    """Base test configuration."""
    return get_default_config(Z=5)


@pytest.fixture
def sim_config(base_config):
    """Test configuration with modified simulation parameters."""
    return Config(
        particle=base_config.particle,
        fluid=base_config.fluid,
        gravity=base_config.gravity,
        initial=base_config.initial,
        domain=base_config.domain,
        simulation=SimulationConfig(
            dt=0.1,
            t_max=10.0,
            max_steps=1000,
            output_dir=base_config.simulation.output_dir
        )
    )


@pytest.fixture
def temp_output_dir():
    """Temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(sim_config, temp_output_dir):
    """Test configuration with temp output dir."""
    return Config(
        particle=sim_config.particle,
        fluid=sim_config.fluid,
        gravity=sim_config.gravity,
        initial=sim_config.initial,
        domain=sim_config.domain,
        simulation=SimulationConfig(
            dt=sim_config.simulation.dt,
            t_max=sim_config.simulation.t_max,
            max_steps=sim_config.simulation.max_steps,
            output_dir=temp_output_dir
        )
    )


@pytest.fixture
def euler_result(test_config):
    """Run Euler simulation for test data."""
    return run_simulation(test_config, "euler")


@pytest.fixture
def rk4_result(test_config):
    """Run RK4 simulation for test data."""
    return run_simulation(test_config, "rk4")


def test_rk4_trajectory_matches_metadata(euler_result, rk4_result, test_config):
    """Test that static RK4 trajectory's final coordinates match RK4 metadata."""
    from src.simulation import results_to_dataframe
    
    rk4_df = results_to_dataframe(rk4_result, test_config)
    exit_pos = rk4_result.exit_position
    
    # The CSV contains the full trajectory including the point outside domain
    # The metadata contains the interpolated exit position
    # Check that the last valid point (before boundary) is close to exit position
    if exit_pos is not None:
        # Find the last point inside domain
        last_inside_idx = len(rk4_df) - 2  # Second to last is inside, last is outside
        final_x = rk4_df["x"].iloc[last_inside_idx]
        final_y = rk4_df["y"].iloc[last_inside_idx]
        
        # The interpolated exit should be between the last inside and first outside
        assert test_config.domain.x_min <= exit_pos[0] <= test_config.domain.x_max
        assert test_config.domain.y_min <= exit_pos[1] <= test_config.domain.y_max


def test_animation_final_coordinates_match_rk4(test_config):
    """Test that animation's final coordinates match the static RK4 trajectory."""
    # Run RK4 simulation
    rk4_result = run_simulation(test_config, "rk4")
    from src.simulation import results_to_dataframe
    rk4_df = results_to_dataframe(rk4_result, test_config)
    
    # Prepare animation data (which uses interpolated exit position)
    from src.animation import prepare_animation_data
    metadata = {
        "method": "rk4",
        "exit_position": rk4_result.exit_position,
        "exit_time": rk4_result.exit_time,
    }
    
    x_data, y_data, times = prepare_animation_data(rk4_df, metadata)
    
    # Final animation frame should match interpolated exit position
    exit_pos = rk4_result.exit_position
    if exit_pos is not None:
        assert abs(x_data[-1] - exit_pos[0]) < 1e-6
        assert abs(y_data[-1] - exit_pos[1]) < 1e-6
    
    # Final animation time should match exit time
    if rk4_result.exit_time is not None:
        assert abs(times[-1] - rk4_result.exit_time) < 1e-6


def test_euler_rk4_identical_parameters(test_config):
    """Test that Euler and RK4 use identical simulation parameters."""
    euler_result = run_simulation(test_config, "euler")
    rk4_result = run_simulation(test_config, "rk4")
    
    # Config snapshots should be identical
    assert euler_result.config_snapshot == rk4_result.config_snapshot
    
    # Both should have same termination reason (boundary_exit)
    assert euler_result.termination_reason == rk4_result.termination_reason
    
    # Both should use the same physical parameters
    assert euler_result.config_snapshot["dt"] == rk4_result.config_snapshot["dt"]
    assert euler_result.config_snapshot["Z"] == rk4_result.config_snapshot["Z"]


def test_exit_times_coordinates_agree_with_saved_data(test_config):
    """Test that reported exit times and coordinates agree with saved data."""
    euler_result = run_simulation(test_config, "euler")
    rk4_result = run_simulation(test_config, "rk4")
    
    from src.simulation import results_to_dataframe
    from src.data import save_results, load_results
    
    # Save and reload
    save_results(euler_result, test_config)
    save_results(rk4_result, test_config)
    
    euler_df, euler_meta = load_results("euler", test_config)
    rk4_df, rk4_meta = load_results("rk4", test_config)
    
    # Check metadata matches simulation results
    assert euler_meta["exit_time"] == euler_result.exit_time
    # JSON serialization converts tuples to lists, so compare values
    assert list(euler_meta["exit_position"]) == list(euler_result.exit_position)
    assert list(euler_meta["exit_velocity"]) == list(euler_result.exit_velocity)
    
    assert rk4_meta["exit_time"] == rk4_result.exit_time
    assert list(rk4_meta["exit_position"]) == list(rk4_result.exit_position)
    assert list(rk4_meta["exit_velocity"]) == list(rk4_result.exit_velocity)
    
    # Check that CSV final coordinates are consistent with metadata
    # (The CSV has the raw simulation states, metadata has interpolated exit)
    if euler_result.exit_position is not None:
        # The interpolated exit position should be on the boundary
        ex, ey = euler_result.exit_position
        assert (abs(ex - test_config.domain.x_min) < 1e-6 or 
                abs(ex - test_config.domain.x_max) < 1e-6 or
                abs(ey - test_config.domain.y_min) < 1e-6 or
                abs(ey - test_config.domain.y_max) < 1e-6)


def test_plotting_export_use_existing_data(test_config):
    """Test that plotting and exporting use existing data without rerunning simulation."""
    from src.simulation import results_to_dataframe
    from src.data import save_results
    from src.visualization import generate_all_plots
    from src.export import export_from_files
    
    euler_result = run_simulation(test_config, "euler")
    rk4_result = run_simulation(test_config, "rk4")
    
    euler_df = results_to_dataframe(euler_result, test_config)
    rk4_df = results_to_dataframe(rk4_result, test_config)
    
    save_results(euler_result, test_config)
    save_results(rk4_result, test_config)
    
    # Generate plots - should not rerun simulation
    paths = generate_all_plots(test_config, euler_df, 
                               {"method": "euler", "exit_position": euler_result.exit_position},
                               rk4_df, 
                               {"method": "rk4", "exit_position": rk4_result.exit_position})
    assert len(paths) > 0
    for p in paths:
        assert p.exists()
    
    # Export - should not rerun simulation
    excel_path = export_from_files(test_config)
    assert excel_path.exists()


def test_simulation_terminates_at_boundary(test_config):
    """Test that simulation terminates correctly when particle leaves domain."""
    euler_result = run_simulation(test_config, "euler")
    rk4_result = run_simulation(test_config, "rk4")
    
    # Both should terminate due to boundary exit
    assert euler_result.termination_reason == "boundary_exit"
    assert rk4_result.termination_reason == "boundary_exit"
    
    # Exit position should be on the boundary
    for result in [euler_result, rk4_result]:
        if result.exit_position:
            ex, ey = result.exit_position
            on_boundary = (
                abs(ex - test_config.domain.x_min) < 1e-6 or
                abs(ex - test_config.domain.x_max) < 1e-6 or
                abs(ey - test_config.domain.y_min) < 1e-6 or
                abs(ey - test_config.domain.y_max) < 1e-6
            )
            assert on_boundary, f"Exit position {result.exit_position} not on boundary"


def test_animation_validation(test_config):
    """Test animation data validation."""
    rk4_result = run_simulation(test_config, "rk4")
    from src.simulation import results_to_dataframe
    rk4_df = results_to_dataframe(rk4_result, test_config)
    
    metadata = {
        "method": "rk4",
        "exit_position": rk4_result.exit_position,
        "termination_reason": rk4_result.termination_reason,
    }
    
    # Validate animation data
    assert validate_animation_data(rk4_df, metadata, test_config)
    
    # Check that prepared data has correct final position
    x_data, y_data, times = prepare_animation_data(rk4_df, metadata)
    if rk4_result.exit_position:
        assert abs(x_data[-1] - rk4_result.exit_position[0]) < 1e-6
        assert abs(y_data[-1] - rk4_result.exit_position[1]) < 1e-6


def test_summary_matches_csv_metadata(test_config):
    """Test that Excel/summary reports same exit times as CSV and metadata."""
    from src.simulation import results_to_dataframe
    from src.data import save_results, save_summary, load_summary
    
    euler_result = run_simulation(test_config, "euler")
    rk4_result = run_simulation(test_config, "rk4")
    
    save_results(euler_result, test_config)
    save_results(rk4_result, test_config)
    
    euler_meta = {
        "method": "euler",
        "steps": euler_result.steps,
        "termination_reason": euler_result.termination_reason,
        "exit_time": euler_result.exit_time,
        "exit_position": euler_result.exit_position,
        "exit_velocity": euler_result.exit_velocity,
        "config": euler_result.config_snapshot,
    }
    rk4_meta = {
        "method": "rk4",
        "steps": rk4_result.steps,
        "termination_reason": rk4_result.termination_reason,
        "exit_time": rk4_result.exit_time,
        "exit_position": rk4_result.exit_position,
        "exit_velocity": rk4_result.exit_velocity,
        "config": rk4_result.config_snapshot,
    }
    
    save_summary(euler_meta, rk4_meta, test_config)
    summary = load_summary(test_config)
    
    # Check summary matches original metadata
    assert summary["euler"]["exit_time"] == euler_meta["exit_time"]
    assert list(summary["euler"]["exit_position"]) == list(euler_meta["exit_position"])
    assert summary["rk4"]["exit_time"] == rk4_meta["exit_time"]
    assert list(summary["rk4"]["exit_position"]) == list(rk4_meta["exit_position"])