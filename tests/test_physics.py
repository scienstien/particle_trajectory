"""Tests for physics module."""

import math
import pytest
import numpy as np

from src.config import Config, get_default_config
from src.physics import (
    particle_mass,
    fluid_velocity,
    relative_velocity,
    drag_force,
    buoyancy_force,
    particle_acceleration,
    state_derivative,
    compute_diagnostics,
    is_outside_domain,
    ParticleState,
)


@pytest.fixture
def config():
    """Default test configuration with Z=5."""
    return get_default_config(Z=5)


def test_particle_mass(config):
    """Test particle mass calculation."""
    m = particle_mass(config.particle)
    expected = (4.0/3.0) * math.pi * config.particle.radius**3 * config.particle.density
    assert abs(m - expected) < 1e-12


def test_fluid_velocity():
    """Test fluid velocity at known coordinates."""
    # At origin
    ux, uy = fluid_velocity(0.0, 0.0)
    assert ux == 1.0
    assert uy == 0.0
    
    # At (1, 0)
    ux, uy = fluid_velocity(1.0, 0.0)
    assert ux == 1.0
    assert abs(uy - math.sin(1.0)) < 1e-12
    
    # At (0, 1)
    ux, uy = fluid_velocity(0.0, 1.0)
    assert ux == 1.0
    assert abs(uy - math.sin(1.0)) < 1e-12


def test_relative_velocity():
    """Test relative velocity calculation."""
    u_rel_x, u_rel_y = relative_velocity(0.5, 0.2, 1.0, 0.5)
    assert u_rel_x == 0.5
    assert u_rel_y == 0.3


def test_drag_force_direction(config):
    """Test drag force is in direction of relative velocity."""
    # Positive relative velocity
    fx, fy = drag_force(1.0, 0.0, config.fluid, config.particle.radius)
    assert fx > 0
    assert fy == 0
    
    # Negative relative velocity
    fx, fy = drag_force(-1.0, 0.0, config.fluid, config.particle.radius)
    assert fx < 0
    assert fy == 0
    
    # Zero relative velocity
    fx, fy = drag_force(0.0, 0.0, config.fluid, config.particle.radius)
    assert fx == 0
    assert fy == 0


def test_buoyancy_force_sign(config):
    """Test buoyancy force sign follows assignment convention (rho_f - rho_p)."""
    # rho_f = 1000, rho_p = 100, so rho_f - rho_p = 900 > 0
    # g_y = 2*(5-5) = 0 for Z=5
    fx, fy = buoyancy_force(config.fluid, config.particle, config.gravity)
    
    # With Z=5, g_y = 0, so force should be zero
    assert fx == 0
    assert fy == 0
    
    # Test with Z=9 (g_y = 8)
    config_z9 = get_default_config(Z=9)
    fx, fy = buoyancy_force(config_z9.fluid, config_z9.particle, config_z9.gravity)
    assert fx == 0  # Horizontal component always zero
    assert fy > 0   # Positive because rho_f > rho_p and g_y > 0


def test_particle_acceleration(config):
    """Test particle acceleration calculation."""
    from src.physics import Forces
    
    mass = particle_mass(config.particle)
    forces = Forces(drag_x=1.0, drag_y=2.0, buoyancy_x=0.5, buoyancy_y=-0.5)
    acc = particle_acceleration(forces, mass)
    
    assert abs(acc.a_x - 1.5/mass) < 1e-12
    assert abs(acc.a_y - 1.5/mass) < 1e-12


def test_state_derivative(config):
    """Test state derivative function."""
    state = np.array([-3.0, 0.0, 0.0, 0.0])
    deriv = state_derivative(0.0, state, config)
    
    # dx/dt = u_x = 0
    assert deriv[0] == 0.0
    # dy/dt = u_y = 0
    assert deriv[1] == 0.0
    # du_x/dt = a_x
    # du_y/dt = a_y
    assert deriv[2] != 0 or deriv[3] != 0  # Some acceleration expected


def test_compute_diagnostics(config):
    """Test diagnostic computation."""
    state = np.array([-3.0, 0.0, 0.0, 0.0])
    diag = compute_diagnostics(state, config)
    
    # Check all expected keys present
    expected_keys = [
        "fluid_u_x", "fluid_u_y", "rel_u_x", "rel_u_y",
        "drag_x", "drag_y", "buoyancy_x", "buoyancy_y",
        "acc_x", "acc_y",
        "dist_x_min", "dist_x_max", "dist_y_min", "dist_y_max"
    ]
    for key in expected_keys:
        assert key in diag


def test_is_outside_domain(config):
    """Test domain boundary detection."""
    # Inside
    state = np.array([0.0, 0.0, 0.0, 0.0])
    assert not is_outside_domain(state, config)
    
    # Outside x_max
    state = np.array([4.0, 0.0, 0.0, 0.0])
    assert is_outside_domain(state, config)
    
    # Outside y_min
    state = np.array([0.0, -4.0, 0.0, 0.0])
    assert is_outside_domain(state, config)


def test_particle_state_conversion():
    """Test ParticleState array conversion."""
    state = ParticleState(1.0, 2.0, 3.0, 4.0)
    arr = state.to_array()
    assert np.array_equal(arr, [1.0, 2.0, 3.0, 4.0])
    
    state2 = ParticleState.from_array(arr)
    assert state2.x == 1.0
    assert state2.y == 2.0
    assert state2.u_x == 3.0
    assert state2.u_y == 4.0


def test_initial_y_calculation():
    """Test initial y is calculated from gravity."""
    config = get_default_config(Z=9)
    # g_y = 2*(9-5) = 8
    # y = -(g_y + 1)/3 = -9/3 = -3
    assert config.initial.y == -3.0
    
    config = get_default_config(Z=0)
    # g_y = 2*(0-5) = -10
    # y = -(-10 + 1)/3 = 9/3 = 3
    assert config.initial.y == 3.0