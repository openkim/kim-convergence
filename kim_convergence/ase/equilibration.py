"""Convergence-controlled equilibration for ASE molecular dynamics.

This module provides utilities for running equilibration phases with
automatic convergence detection using kim-convergence.
"""

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from kim_convergence import run_length_control

from .extractors import DEFAULT_EXTRACTORS

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from ase.md.md import MolecularDynamics

__all__ = ["ASESampler", "run_ase_equilibration"]


class ASESampler:
    """Trajectory sampler for ASE molecular dynamics.

    This sampler advances the MD simulation and collects property values
    for convergence analysis using kim-convergence. It uses ASE's observer
    mechanism for efficient data collection.

    Examples:
        >>> from ase.md.langevin import Langevin
        >>> from kim_convergence.ase import ASESampler, run_ase_equilibration
        >>>
        >>> dyn = Langevin(atoms, timestep=1.0, temperature_K=300, friction=0.01)
        >>> sampler = ASESampler(dyn, property_name="temperature")
        >>> result = run_ase_equilibration(sampler, maximum_run_length=10000)
    """

    def __init__(
        self,
        dynamics: "MolecularDynamics",
        property_name: str = "energy",
        sample_interval: int = 1,
        extractors: Optional[Dict[str, Callable[["Atoms"], float]]] = None,
    ):
        """Initialize the sampler.

        Args:
            dynamics: ASE MolecularDynamics object.
            property_name: Name of the property to monitor. Available options:
                "energy" (or "potential_energy"), "kinetic_energy", "total_energy",
                "volume", "pressure", "temperature", "density".
                Default: "energy".
            sample_interval: Collect property every N steps. Default: 1.
            extractors: Optional dictionary of custom property extractors.
                Keys are property names, values are functions that take an
                Atoms object and return a float.
        """
        self.dynamics = dynamics
        self.property_name = property_name
        self.sample_interval = max(1, sample_interval)
        self._total_steps = 0

        # Set up property extractors
        self.extractors = DEFAULT_EXTRACTORS.copy()
        if extractors:
            self.extractors.update(extractors)

        # Validate that the requested property has an extractor
        if self.property_name not in self.extractors:
            available = ", ".join(sorted(self.extractors.keys()))
            raise ValueError(
                f"No extractor available for property: {self.property_name}. "
                f"Available properties: {available}"
            )

        self._extractor = self.extractors[self.property_name]

    def __call__(self, nstep: int) -> np.ndarray:
        """Advance the dynamics nstep steps and return property values.

        Uses ASE's observer mechanism for efficient batch execution.

        Args:
            nstep: Number of samples to collect. The actual number of MD steps
                run will be nstep * sample_interval.

        Returns:
            1D array of nstep + 1 property values collected at each sample_interval.
        """
        md_steps = nstep * self.sample_interval
        property_values: List[float] = []

        def collect_property() -> None:
            """Observer callback to collect property value."""
            value = self._extractor(self.dynamics.atoms)
            property_values.append(value)

        # Attach observer for data collection
        self.dynamics.attach(collect_property, interval=self.sample_interval)

        try:
            # Run all steps at once (much faster than run(1) in a loop)
            self.dynamics.run(md_steps)
            self._total_steps += md_steps
        finally:
            # Clean up: remove our observer
            for i, (func, *_) in enumerate(self.dynamics.observers):
                if func is collect_property:
                    self.dynamics.observers.pop(i)
                    break

        return np.asarray(property_values, dtype=np.float64)

    @property
    def total_steps(self) -> int:
        """Return the total number of MD steps run so far."""
        return self._total_steps


def run_ase_equilibration(
    sampler: ASESampler,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run convergence-controlled equilibration for ASE molecular dynamics.

    This function runs an equilibration phase using kim-convergence to
    automatically detect when the system has reached equilibrium.

    Args:
        sampler: An ASESampler instance that will be used to collect
            trajectory data during the simulation.
        **kwargs: Keyword arguments passed to kim_convergence.run_length_control().
            Common options include:
            - initial_run_length (int): Initial samples before checking. Default: 10000.
            - maximum_run_length (int): Maximum samples to collect. Default: 1000000.
            - maximum_equilibration_step (int): Max samples for equilibration detection.
            - relative_accuracy (float): Target relative accuracy. Default: 0.1.
            - absolute_accuracy (float): Target absolute accuracy. Default: 0.1.
            - confidence_coefficient (float): Confidence level. Default: 0.95.
            See kim_convergence.run_length_control for all available options.

    Returns:
        Dictionary containing the kim-convergence run_length_control result.
        Key fields include:
        - converged (bool): Whether convergence was achieved.
        - total_run_length (int): Total number of samples collected.
        - equilibration_step (int): Sample at which equilibration was detected.
        - mean (float): Estimated mean of the monitored property.
        - standard_deviation (float): Standard deviation.
        See kim_convergence.run_length_control documentation for full details.

    Examples:
        Basic usage:

        >>> from ase.build import bulk
        >>> from ase.calculators.emt import EMT
        >>> from ase.md.langevin import Langevin
        >>> from ase import units
        >>> from kim_convergence.ase import ASESampler, run_ase_equilibration
        >>>
        >>> atoms = bulk('Cu', cubic=True) * (3, 3, 3)
        >>> atoms.calc = EMT()
        >>> dyn = Langevin(atoms, 5 * units.fs, temperature_K=500, friction=0.01)
        >>>
        >>> sampler = ASESampler(dyn, property_name="temperature")
        >>> result = run_ase_equilibration(
        ...     sampler,
        ...     initial_run_length=1000,
        ...     maximum_run_length=10000,
        ...     relative_accuracy=0.1,
        ... )
        >>> if result["converged"]:
        ...     print(f"Equilibrated! Mean T = {result['mean']:.1f} K")

        With sample_interval to reduce data collection:

        >>> sampler = ASESampler(dyn, property_name="energy", sample_interval=10)
        >>> result = run_ase_equilibration(
        ...     sampler,
        ...     maximum_run_length=5000,  # 5000 samples = 50000 MD steps
        ...     relative_accuracy=0.05,
        ... )

        With custom property extractor:

        >>> def get_max_force(atoms):
        ...     return np.max(np.abs(atoms.get_forces()))
        >>>
        >>> sampler = ASESampler(
        ...     dyn,
        ...     property_name="max_force",
        ...     extractors={"max_force": get_max_force},
        ... )
        >>> result = run_ase_equilibration(sampler, relative_accuracy=0.1)
    """
    # Build kwargs for run_length_control
    # Prevent override of get_trajectory (sampler) as it's critical to function operation
    if "get_trajectory" in kwargs:
        raise ValueError(
            "Cannot override 'get_trajectory' parameter. "
            "The sampler is automatically set from the provided sampler argument."
        )
    
    rlc_kwargs: Dict[str, Any] = {
        "get_trajectory": sampler,
        "number_of_variables": 1,
        "fp": "return",
        "fp_format": "json",
    }
    rlc_kwargs.update(kwargs)
    # Reassert get_trajectory after merging to ensure it cannot be overridden
    rlc_kwargs["get_trajectory"] = sampler

    # Run convergence control
    result_json = run_length_control(**rlc_kwargs)

    return json.loads(result_json)
