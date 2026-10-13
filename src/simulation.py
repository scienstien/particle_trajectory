"""Simulation loop for Lagrangian Particle Tracking."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import Config
from .physics import (
    ParticleState,
    compute_diagnostics,
    is_outside_domain,
    state_derivative,
)
from .integrators import integrate


@dataclass(frozen=True)
class SimulationResult:
    """Complete simulation results."""
    method: str
    states: list[np.ndarray]          # List of state arrays
    times: list[float]                # List of time values
    steps: int                        # Number of integration steps
    termination_reason: str           # "boundary_exit", "max_time", "max_steps"
    exit_time: float | None           # Interpolated exit time (if boundary_exit)
    exit_position: tuple[float, float] | None  # Interpolated exit position
    exit_velocity: tuple[float, float] | None  # Velocity at exit
    config_snapshot: dict             # Configuration used


def interpolate_boundary_crossing(
    prev_state: np.ndarray,
    curr_state: np.ndarray,
    prev_time: float,
    curr_time: float,
    config: Config
) -> tuple[float, tuple[float, float]]:
    """
    Estimate boundary crossing time and position by linear interpolation.
    
    Returns (exit_time, (exit_x, exit_y)).
    """
    # Find which boundary was crossed
    x0, y0 = prev_state[0], prev_state[1]
    x1, y1 = curr_state[0], curr_state[1]
    
    # Check each boundary for crossing
    t_cross = None
    x_cross = None
    y_cross = None
    
    # Check x_min boundary
    if x0 >= config.domain.x_min and x1 < config.domain.x_min:
        alpha = (config.domain.x_min - x0) / (x1 - x0) if x1 != x0 else 0
        t_cross = prev_time + alpha * (curr_time - prev_time)
        x_cross = config.domain.x_min
        y_cross = y0 + alpha * (y1 - y0)
    
    # Check x_max boundary
    elif x0 <= config.domain.x_max and x1 > config.domain.x_max:
        alpha = (config.domain.x_max - x0) / (x1 - x0) if x1 != x0 else 0
        t_cross = prev_time + alpha * (curr_time - prev_time)
        x_cross = config.domain.x_max
        y_cross = y0 + alpha * (y1 - y0)
    
    # Check y_min boundary
    elif y0 >= config.domain.y_min and y1 < config.domain.y_min:
        alpha = (config.domain.y_min - y0) / (y1 - y0) if y1 != y0 else 0
        t_cross = prev_time + alpha * (curr_time - prev_time)
        x_cross = x0 + alpha * (x1 - x0)
        y_cross = config.domain.y_min
    
    # Check y_max boundary
    elif y0 <= config.domain.y_max and y1 > config.domain.y_max:
        alpha = (config.domain.y_max - y0) / (y1 - y0) if y1 != y0 else 0
        t_cross = prev_time + alpha * (curr_time - prev_time)
        x_cross = x0 + alpha * (x1 - x0)
        y_cross = config.domain.y_max
    
    if t_cross is None:
        # Fallback: use current time and position
        return curr_time, (x1, y1)
    
    return t_cross, (x_cross, y_cross)


def run_simulation(
    config: Config,
    method: Literal["euler", "rk4"]
) -> SimulationResult:
    """
    Run a single simulation with the given integration method.
    
    Args:
        config: Simulation configuration
        method: Integration method ("euler" or "rk4")
    
    Returns:
        SimulationResult with all recorded data
    """
    # Initial state
    initial_state = np.array([
        config.initial.x,
        config.initial.y,
        config.initial.u_x,
        config.initial.u_y
    ], dtype=float)
    
    # Check if initial state is already outside domain
    if is_outside_domain(initial_state, config):
        return SimulationResult(
            method=method,
            states=[initial_state],
            times=[0.0],
            steps=0,
            termination_reason="initial_outside",
            exit_time=0.0,
            exit_position=(config.initial.x, config.initial.y),
            exit_velocity=(config.initial.u_x, config.initial.u_y),
            config_snapshot=config_to_dict(config)
        )
    
    # Derivative function bound to config
    def f(t: float, state: np.ndarray) -> np.ndarray:
        return state_derivative(t, state, config)
    
    # Stop condition: particle leaves domain
    def stop_condition(state: np.ndarray) -> bool:
        return is_outside_domain(state, config)
    
    # Run integration
    states, times, steps, termination_reason = integrate(
        f=f,
        initial_state=initial_state,
        dt=config.simulation.dt,
        max_steps=config.simulation.max_steps,
        t_max=config.simulation.t_max,
        method=method,
        stop_condition=stop_condition
    )
    
    # Handle boundary exit interpolation
    exit_time = None
    exit_position = None
    exit_velocity = None
    
    if termination_reason == "boundary_exit" and len(states) >= 2:
        # The last state is outside, second-to-last is inside
        exit_time, exit_position = interpolate_boundary_crossing(
            states[-2], states[-1], times[-2], times[-1], config
        )
        exit_velocity = (states[-2][2], states[-2][3])  # Velocity at last inside step
    elif termination_reason == "boundary_exit" and len(states) == 1:
        # Edge case: somehow exited in 0 steps
        exit_time = 0.0
        exit_position = (states[0][0], states[0][1])
        exit_velocity = (states[0][2], states[0][3])
    elif termination_reason in ("max_time", "max_steps"):
        # Particle didn't exit - use final state
        exit_time = times[-1]
        exit_position = (states[-1][0], states[-1][1])
        exit_velocity = (states[-1][2], states[-1][3])
    
    return SimulationResult(
        method=method,
        states=states,
        times=times,
        steps=steps,
        termination_reason=termination_reason,
        exit_time=exit_time,
        exit_position=exit_position,
        exit_velocity=exit_velocity,
        config_snapshot=config_to_dict(config)
    )


def config_to_dict(config: Config) -> dict:
    """Convert config to a serializable dictionary."""
    return {
        "Z": config.gravity.Z,
        "g_y": config.gravity.g_y,
        "particle_radius": config.particle.radius,
        "particle_density": config.particle.density,
        "fluid_density": config.fluid.density,
        "drag_coefficient": config.fluid.drag_coefficient,
        "initial_x": config.initial.x,
        "initial_y": config.initial.y,
        "initial_u_x": config.initial.u_x,
        "initial_u_y": config.initial.u_y,
        "domain_x_min": config.domain.x_min,
        "domain_x_max": config.domain.x_max,
        "domain_y_min": config.domain.y_min,
        "domain_y_max": config.domain.y_max,
        "dt": config.simulation.dt,
        "t_max": config.simulation.t_max,
        "max_steps": config.simulation.max_steps,
    }


def results_to_dataframe(result: SimulationResult, config: Config) -> pd.DataFrame:
    """Convert simulation results to a pandas DataFrame with all diagnostics."""
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


def run_both_methods(config: Config) -> tuple[SimulationResult, SimulationResult]:
    """Run both Euler and RK4 simulations."""
    euler_result = run_simulation(config, "euler")
    rk4_result = run_simulation(config, "rk4")
    return euler_result, rk4_result