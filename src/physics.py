"""Physics calculations for Lagrangian Particle Tracking."""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import Config, ParticleConfig, FluidConfig, GravityConfig


@dataclass(frozen=True)
class ParticleState:
    """Particle state vector: [x, y, u_x, u_y]."""
    x: float
    y: float
    u_x: float
    u_y: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.u_x, self.u_y], dtype=float)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ParticleState":
        """Create from numpy array."""
        return cls(arr[0], arr[1], arr[2], arr[3])


@dataclass(frozen=True)
class Forces:
    """Force components acting on the particle."""
    drag_x: float
    drag_y: float
    buoyancy_x: float
    buoyancy_y: float
    
    @property
    def net_x(self) -> float:
        return self.drag_x + self.buoyancy_x
    
    @property
    def net_y(self) -> float:
        return self.drag_y + self.buoyancy_y


@dataclass(frozen=True)
class Acceleration:
    """Particle acceleration components."""
    a_x: float
    a_y: float


def particle_mass(config: ParticleConfig) -> float:
    """Calculate particle mass: m_p = (4/3) * pi * r_p^3 * rho_p."""
    return (4.0 / 3.0) * math.pi * config.radius**3 * config.density


def fluid_velocity(x: float, y: float) -> Tuple[float, float]:
    """Fluid velocity field: u_fluid = (1, sin(x^2 + y^2))."""
    return (1.0, math.sin(x**2 + y**2))


def relative_velocity(
    u_particle_x: float, u_particle_y: float,
    u_fluid_x: float, u_fluid_y: float
) -> Tuple[float, float]:
    """Relative velocity: u_rel = u_fluid - u_particle."""
    return (u_fluid_x - u_particle_x, u_fluid_y - u_particle_y)


def drag_force(
    u_rel_x: float, u_rel_y: float,
    config: FluidConfig, particle_radius: float
) -> Tuple[float, float]:
    """
    Drag force: F_drag = 0.5 * C_D * rho_f * pi * r_p^2 * |u_rel| * u_rel
    
    The force is in the direction of relative velocity.
    """
    u_rel_mag = math.hypot(u_rel_x, u_rel_y)
    if u_rel_mag == 0:
        return (0.0, 0.0)
    
    coeff = 0.5 * config.drag_coefficient * config.density * math.pi * particle_radius**2 * u_rel_mag
    return (coeff * u_rel_x, coeff * u_rel_y)


def buoyancy_force(
    config: FluidConfig, particle_config: ParticleConfig, gravity: GravityConfig
) -> Tuple[float, float]:
    """
    Buoyancy/gravity force: F_buoyancy = (rho_f - rho_p) * (4/3) * pi * r_p^3 * g_vector
    
    IMPORTANT: Uses (rho_f - rho_p) as per assignment convention.
    This is NOT the conventional expression which would use (rho_p - rho_f).
    """
    coeff = (config.density - particle_config.density) * (4.0 / 3.0) * math.pi * particle_config.radius**3
    g_x, g_y = gravity.g_vector
    return (coeff * g_x, coeff * g_y)


def particle_acceleration(
    forces: Forces, mass: float
) -> Acceleration:
    """Net acceleration: a_particle = (F_drag + F_buoyancy) / m_p."""
    return Acceleration(forces.net_x / mass, forces.net_y / mass)


def state_derivative(t: float, state: np.ndarray, config: Config) -> np.ndarray:
    """
    Full state derivative function for integration.
    
    State: [x, y, u_x_particle, u_y_particle]
    Returns: [dx/dt, dy/dt, du_x/dt, du_y/dt] = [u_x, u_y, a_x, a_y]
    """
    x, y, u_x, u_y = state
    
    # Fluid velocity at particle position
    u_fluid_x, u_fluid_y = fluid_velocity(x, y)
    
    # Relative velocity
    u_rel_x, u_rel_y = relative_velocity(u_x, u_y, u_fluid_x, u_fluid_y)
    
    # Forces
    mass = particle_mass(config.particle)
    drag_x, drag_y = drag_force(u_rel_x, u_rel_y, config.fluid, config.particle.radius)
    buoy_x, buoy_y = buoyancy_force(config.fluid, config.particle, config.gravity)
    
    # Acceleration
    a_x = (drag_x + buoy_x) / mass
    a_y = (drag_y + buoy_y) / mass
    
    return np.array([u_x, u_y, a_x, a_y], dtype=float)


def compute_diagnostics(
    state: np.ndarray, config: Config
) -> dict:
    """Compute all diagnostic quantities for a given state."""
    x, y, u_x, u_y = state
    
    # Fluid velocity at particle position
    u_fluid_x, u_fluid_y = fluid_velocity(x, y)
    
    # Relative velocity
    u_rel_x, u_rel_y = relative_velocity(u_x, u_y, u_fluid_x, u_fluid_y)
    
    # Forces
    mass = particle_mass(config.particle)
    drag_x, drag_y = drag_force(u_rel_x, u_rel_y, config.fluid, config.particle.radius)
    buoy_x, buoy_y = buoyancy_force(config.fluid, config.particle, config.gravity)
    
    # Acceleration
    a_x = (drag_x + buoy_x) / mass
    a_y = (drag_y + buoy_y) / mass
    
    # Distance to boundaries
    dist_x_min = x - config.domain.x_min
    dist_x_max = config.domain.x_max - x
    dist_y_min = y - config.domain.y_min
    dist_y_max = config.domain.y_max - y
    
    return {
        "fluid_u_x": u_fluid_x,
        "fluid_u_y": u_fluid_y,
        "rel_u_x": u_rel_x,
        "rel_u_y": u_rel_y,
        "drag_x": drag_x,
        "drag_y": drag_y,
        "buoyancy_x": buoy_x,
        "buoyancy_y": buoy_y,
        "acc_x": a_x,
        "acc_y": a_y,
        "dist_x_min": dist_x_min,
        "dist_x_max": dist_x_max,
        "dist_y_min": dist_y_min,
        "dist_y_max": dist_y_max,
    }


def is_outside_domain(state: np.ndarray, config: Config) -> bool:
    """Check if particle is outside the simulation domain."""
    x, y = state[0], state[1]
    return (x < config.domain.x_min or x > config.domain.x_max or
            y < config.domain.y_min or y > config.domain.y_max)