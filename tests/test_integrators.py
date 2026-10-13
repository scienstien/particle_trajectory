"""Tests for integrators module."""

import numpy as np
import pytest

from src.integrators import euler_step, rk4_step, integrate


def test_euler_step_simple():
    """Test Euler step on simple derivative."""
    # dy/dt = y, exact solution y(t) = exp(t)
    def f(t, y):
        return y
    
    y0 = np.array([1.0])
    dt = 0.1
    y1 = euler_step(f, 0.0, y0, dt)
    
    # Euler: y1 = y0 + dt * y0 = 1 + 0.1*1 = 1.1
    assert abs(y1[0] - 1.1) < 1e-12


def test_rk4_step_simple():
    """Test RK4 step on simple derivative."""
    # dy/dt = y, exact solution y(t) = exp(t)
    def f(t, y):
        return y
    
    y0 = np.array([1.0])
    dt = 0.1
    y1 = rk4_step(f, 0.0, y0, dt)
    
    # RK4 should be more accurate than Euler
    # Exact: exp(0.1) ≈ 1.105170918
    # Euler: 1.1
    # RK4 error is O(dt^5) ~ 1e-5 for dt=0.1
    exact = np.exp(0.1)
    assert abs(y1[0] - exact) < 1e-7


def test_rk4_accuracy_exponential():
    """Test RK4 accuracy on exponential ODE over multiple steps."""
    def f(t, y):
        return y
    
    y0 = np.array([1.0])
    dt = 0.01
    t_max = 1.0
    steps = int(t_max / dt)
    
    y = y0.copy()
    t = 0.0
    for _ in range(steps):
        y = rk4_step(f, t, y, dt)
        t += dt
    
    exact = np.exp(1.0)
    # RK4 should be very accurate
    assert abs(y[0] - exact) < 1e-8


def test_euler_accuracy_exponential():
    """Test Euler accuracy on exponential ODE (less accurate)."""
    def f(t, y):
        return y
    
    y0 = np.array([1.0])
    dt = 0.01
    t_max = 1.0
    steps = int(t_max / dt)
    
    y = y0.copy()
    t = 0.0
    for _ in range(steps):
        y = euler_step(f, t, y, dt)
        t += dt
    
    exact = np.exp(1.0)
    # Euler error should be larger than RK4
    assert abs(y[0] - exact) > 1e-3


def test_rk4_harmonic_oscillator():
    """Test RK4 on harmonic oscillator: d2x/dt2 = -x."""
    # State: [x, v]
    def f(t, state):
        x, v = state
        return np.array([v, -x])
    
    # Exact solution: x = cos(t), v = -sin(t)
    state = np.array([1.0, 0.0])  # x=1, v=0
    dt = 0.01
    t_max = 2 * np.pi
    steps = int(t_max / dt)
    
    for i in range(steps):
        t = i * dt
        state = rk4_step(f, t, state, dt)
    
    # After one period, should return close to initial state
    # Error accumulates over many steps (628 steps)
    assert abs(state[0] - 1.0) < 1e-5
    assert abs(state[1] - 0.0) < 5e-3


def test_integrate_with_stop_condition():
    """Test integration loop with stop condition."""
    def f(t, y):
        return np.array([1.0])  # Constant velocity
    
    def stop(state):
        return state[0] >= 5.0
    
    states, times, steps, reason = integrate(
        f, np.array([0.0]), 0.1, 1000, 100.0, "euler", stop
    )
    
    assert reason == "boundary_exit"
    assert steps > 0
    assert states[-1][0] >= 5.0


def test_integrate_max_time():
    """Test integration stops at max time."""
    def f(t, y):
        return np.array([1.0])
    
    states, times, steps, reason = integrate(
        f, np.array([0.0]), 0.1, 1000, 1.0, "euler", None
    )
    
    assert reason == "max_time"
    assert abs(times[-1] - 1.0) < 1e-10


def test_integrate_max_steps():
    """Test integration stops at max steps."""
    def f(t, y):
        return np.array([1.0])
    
    states, times, steps, reason = integrate(
        f, np.array([0.0]), 0.1, 10, 100.0, "euler", None
    )
    
    assert reason == "max_steps"
    assert steps == 10


def test_integrate_method_selection():
    """Test that method selection works."""
    def f(t, y):
        return y
    
    # Euler
    states_e, times_e, _, _ = integrate(
        f, np.array([1.0]), 0.1, 10, 10.0, "euler", None
    )
    
    # RK4
    states_rk4, times_rk4, _, _ = integrate(
        f, np.array([1.0]), 0.1, 10, 10.0, "rk4", None
    )
    
    # RK4 should be more accurate
    exact = np.exp(1.0)
    euler_final = states_e[-1][0]
    rk4_final = states_rk4[-1][0]
    
    assert abs(rk4_final - exact) < abs(euler_final - exact)


def test_integrate_unknown_method():
    """Test error on unknown method."""
    def f(t, y):
        return y
    
    with pytest.raises(ValueError):
        integrate(f, np.array([1.0]), 0.1, 10, 10.0, "unknown", None)