"""Tests for data persistence module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import get_default_config, Config, SimulationConfig
from src.data import (
    save_results,
    load_results,
    validate_dataframe,
    save_summary,
    load_summary,
    check_results_exist,
    get_output_paths,
    REQUIRED_COLUMNS,
)
from src.simulation import run_simulation, results_to_dataframe


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
def sample_result(test_config):
    """Run a quick simulation for test data."""
    return run_simulation(test_config, "euler")


def test_save_and_load_results(test_config, sample_result):
    """Test CSV and metadata save/load round trip."""
    # Save
    csv_path, meta_path = save_results(sample_result, test_config)
    assert csv_path.exists()
    assert meta_path.exists()
    
    # Load
    df, metadata = load_results("euler", test_config)
    
    # Check data integrity
    assert len(df) == len(sample_result.states)
    assert metadata["method"] == "euler"
    assert metadata["steps"] == sample_result.steps
    assert metadata["termination_reason"] == sample_result.termination_reason


def test_validate_dataframe(test_config):
    """Test DataFrame validation."""
    # Valid DataFrame
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    validate_dataframe(df)  # Should not raise
    
    # Missing columns
    df_invalid = pd.DataFrame(columns=["step", "time"])
    with pytest.raises(ValueError):
        validate_dataframe(df_invalid)


def test_load_missing_files(test_config):
    """Test loading non-existent files raises error."""
    with pytest.raises(FileNotFoundError):
        load_results("euler", test_config)


def test_save_summary(test_config, sample_result):
    """Test saving combined summary."""
    # Create minimal metadata
    euler_meta = {
        "method": "euler",
        "steps": sample_result.steps,
        "termination_reason": sample_result.termination_reason,
        "exit_time": sample_result.exit_time,
        "exit_position": sample_result.exit_position,
        "exit_velocity": sample_result.exit_velocity,
        "config": sample_result.config_snapshot,
    }
    rk4_meta = euler_meta.copy()
    rk4_meta["method"] = "rk4"
    
    summary_path = save_summary(euler_meta, rk4_meta, test_config)
    assert summary_path.exists()
    
    # Load and verify
    summary = load_summary(test_config)
    assert "euler" in summary
    assert "rk4" in summary
    assert "config" in summary


def test_check_results_exist(test_config, sample_result):
    """Test checking which results exist."""
    # Initially none exist
    exists = check_results_exist(test_config, ["euler", "rk4"])
    assert not exists["euler"]
    assert not exists["rk4"]
    
    # Save euler only
    save_results(sample_result, test_config)
    exists = check_results_exist(test_config, ["euler", "rk4"])
    assert exists["euler"]
    assert not exists["rk4"]


def test_get_output_paths(test_config):
    """Test output path generation."""
    csv_path, meta_path = get_output_paths(test_config, "euler")
    
    assert csv_path == test_config.simulation.output_dir / "euler_results.csv"
    assert meta_path == test_config.simulation.output_dir / "euler_metadata.json"


def test_results_to_dataframe(test_config, sample_result):
    """Test conversion to DataFrame."""
    df = results_to_dataframe(sample_result, test_config)
    
    assert len(df) == len(sample_result.states)
    assert list(df.columns) == REQUIRED_COLUMNS
    
    # Check first row matches initial state
    assert df["step"].iloc[0] == 0
    assert df["time"].iloc[0] == 0.0
    assert df["x"].iloc[0] == test_config.initial.x
    assert df["y"].iloc[0] == test_config.initial.y


def test_metadata_json_content(test_config, sample_result):
    """Test metadata JSON has correct structure."""
    save_results(sample_result, test_config)
    
    meta_path = test_config.simulation.output_dir / "euler_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    
    assert meta["method"] == "euler"
    assert "steps" in meta
    assert "termination_reason" in meta
    assert "exit_time" in meta
    assert "exit_position" in meta
    assert "exit_velocity" in meta
    assert "config" in meta