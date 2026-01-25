"""ASE integration for kim-convergence.

This module provides utilities for running convergence-controlled
equilibration with ASE (Atomic Simulation Environment) molecular dynamics.

Note:
    This module requires ASE to be installed. Install with:
        pip install ase

Example usage:

    >>> from ase.md.langevin import Langevin
    >>> from kim_convergence.ase import run_ase_equilibration, DEFAULT_EXTRACTORS
    >>>
    >>> dyn = Langevin(atoms, timestep=1.0, temperature_K=300, friction=0.01)
    >>> result = run_ase_equilibration(
    ...     dynamics=dyn,
    ...     property_name="energy",
    ...     maximum_run_length=10000,
    ...     relative_accuracy=0.05,
    ... )
    >>> if result["converged"]:
    ...     print(f"Equilibrated in {result['equilibration_step']} steps")
"""

# Check if ASE is available
try:
    import ase  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The kim_convergence.ase module requires ASE (Atomic Simulation Environment). "
        "Install it with: pip install ase"
    ) from e

from .extractors import (
    DEFAULT_EXTRACTORS,
    get_potential_energy,
    get_kinetic_energy,
    get_total_energy,
    get_volume,
    get_pressure,
    get_temperature,
    get_density,
)
from .equilibration import ASESampler, run_ase_equilibration

__all__ = [
    # Extractors
    "DEFAULT_EXTRACTORS",
    "get_potential_energy",
    "get_kinetic_energy",
    "get_total_energy",
    "get_volume",
    "get_pressure",
    "get_temperature",
    "get_density",
    # Equilibration
    "ASESampler",
    "run_ase_equilibration",
]
