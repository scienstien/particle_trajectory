"""Numerical integrators for ODE systems."""

from typing import Callable

import numpy as np


DerivativeFunc = Callable[[float, np.ndarray], np.ndarray]


def euler_step(
    f: DerivativeFunc,
    t: float,
    state: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Forward Euler step.
    
    state_next = state + dt * f(t, state)
    
    Args:
        f: Derivative function f(t, state) -> dstate/dt
        t: Current time
        state: Current state vector
        dt: Timestep
    
    Returns:
        Next state vector
    """
    return state + dt * f(t, state)


def rk4_step(
    f: DerivativeFunc,
    t: float,
    state: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Classical fourth-order Runge-Kutta step.
    
    k1 = f(t, state)
    k2 = f(t + dt/2, state + dt*k1/2)
    k3 = f(t + dt/2, state + dt*k2/2)
    k4 = f(t + dt, state + dt*k3)
    state_next = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    Args:
        f: Derivative function f(t, state) -> dstate/dt
        t: Current time
        state: Current state vector
        dt: Timestep
    
    Returns:
        Next state vector
    """
    k1 = f(t, state)
    k2 = f(t + dt/2, state + dt * k1 / 2)
    k3 = f(t + dt/2, state + dt * k2 / 2)
    k4 = f(t + dt, state + dt * k3)
    
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def integrate(
    f: DerivativeFunc,
    initial_state: np.ndarray,
    dt: float,
    max_steps: int,
    t_max: float,
    method: str,
    stop_condition: Callable[[np.ndarray], bool] | None = None
) -> tuple[list[np.ndarray], list[float], int, str]:
    """
    Generic integration loop.
    
    Args:
        f: Derivative function
        initial_state: Initial state vector
        dt: Timestep
        max_steps: Maximum number of steps
        t_max: Maximum simulation time
        method: "euler" or "rk4"
        stop_condition: Optional function(state) -> bool to stop early
    
    Returns:
        Tuple of (states_list, times_list, steps_taken, termination_reason)
    """
    if method == "euler":
        step_func = euler_step
    elif method == "rk4":
        step_func = rk4_step
    else:
        raise ValueError(f"Unknown method: {method}")
    
    states = [initial_state.copy()]
    times = [0.0]
    state = initial_state.copy()
    t = 0.0
    
    termination_reason = "max_steps"
    
    for step in range(1, max_steps + 1):
        # Check time limit
        if t >= t_max:
            termination_reason = "max_time"
            break
        
        # Check stop condition (e.g., boundary exit)
        if stop_condition is not None and stop_condition(state):
            termination_reason = "boundary_exit"
            break
        
        # Perform integration step
        state = step_func(f, t, state, dt)
        t = step * dt
        
        states.append(state.copy())
        times.append(t)
        
        # Check stop condition after step
        if stop_condition is not None and stop_condition(state):
            termination_reason = "boundary_exit"
            break
    else:
        termination_reason = "max_steps"
    
    return states, times, len(states) - 1, termination_reason