"""Configuration parameters for the Lagrangian Particle Tracking simulation."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ParticleConfig:
    """Particle physical properties."""
    radius: float = 0.01          # m
    density: float = 100.0        # kg/m^3


@dataclass(frozen=True)
class FluidConfig:
    """Fluid physical properties."""
    density: float = 1000.0       # kg/m^3
    drag_coefficient: float = 0.3


@dataclass(frozen=True)
class GravityConfig:
    """Gravity configuration. Z is the last digit of roll number."""
    Z: int = 0                    # Last digit of roll number (configurable)
    
    @property
    def g_y(self) -> float:
        """Vertical gravity component: g_y = 2 * (Z - 5) m/s^2."""
        return 2.0 * (self.Z - 5)
    
    @property
    def g_vector(self) -> tuple[float, float]:
        """Gravity vector (horizontal component is zero)."""
        return (0.0, self.g_y)


@dataclass(frozen=True)
class InitialConfig:
    """Initial conditions for the particle."""
    x: float = -3.0               # m
    y: float = None               # m (computed from gravity if None)
    u_x: float = 0.0              # m/s
    u_y: float = 0.0              # m/s
    
    def __post_init__(self):
        if self.y is None:
            # y = -(g_y + 1) / 3
            # This will be set after gravity config is known
            pass


@dataclass(frozen=True)
class DomainConfig:
    """Simulation domain boundaries."""
    x_min: float = -3.0
    x_max: float = 3.0
    y_min: float = -3.0
    y_max: float = 3.0


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation control parameters."""
    dt: float = 0.001             # Timestep (s)
    t_max: float = 100.0          # Maximum simulation time (s)
    max_steps: int = 1_000_000    # Maximum number of steps
    output_dir: Path = Path("output")
    run_id: str = ""              # Run identifier (auto-generated if empty)
    
    def get_run_output_dir(self) -> Path:
        """Get the run-specific output directory."""
        if self.run_id:
            return self.output_dir / self.run_id
        return self.output_dir


def generate_run_id(Z: int) -> str:
    """Generate a unique run ID based on timestamp and Z value."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_Z{Z}_{timestamp}"


@dataclass(frozen=True)
class Config:
    """Complete simulation configuration."""
    particle: ParticleConfig = field(default_factory=ParticleConfig)
    fluid: FluidConfig = field(default_factory=FluidConfig)
    gravity: GravityConfig = field(default_factory=GravityConfig)
    initial: InitialConfig = field(default_factory=InitialConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Integration methods to run
    methods: tuple[Literal["euler", "rk4"], ...] = ("euler", "rk4")
    
    def __post_init__(self):
        # Set initial y based on gravity if not explicitly provided
        if self.initial.y is None:
            object.__setattr__(self.initial, 'y', -(self.gravity.g_y + 1) / 3)
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not isinstance(self.gravity.Z, int):
            raise ValueError(f"Z must be an integer, got {self.gravity.Z}")
        if self.particle.radius <= 0:
            raise ValueError("Particle radius must be positive")
        if self.particle.density <= 0:
            raise ValueError("Particle density must be positive")
        if self.fluid.density <= 0:
            raise ValueError("Fluid density must be positive")
        if not 0 < self.fluid.drag_coefficient <= 2:
            raise ValueError("Drag coefficient should be in (0, 2]")
        if self.simulation.dt <= 0:
            raise ValueError("Timestep must be positive")
        if self.simulation.t_max <= 0:
            raise ValueError("Maximum simulation time must be positive")
        if self.simulation.max_steps <= 0:
            raise ValueError("Maximum steps must be positive")
        if self.domain.x_min >= self.domain.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.domain.y_min >= self.domain.y_max:
            raise ValueError("y_min must be less than y_max")


def get_default_config(Z: int = 0) -> Config:
    """Create a default configuration with the given Z value."""
    config = Config(gravity=GravityConfig(Z=Z))
    config.validate()
    return config