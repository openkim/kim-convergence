"""Convergence-controlled equilibration for ASE molecular dynamics.

This module provides utilities for running equilibration phases with
automatic convergence detection using kim-convergence.
"""

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from kim_convergence import run_length_control, CRError, cr_check

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
        >>> sampler = ASESampler(dyn, property_names="temperature")
        >>> result = run_ase_equilibration(sampler, maximum_run_length=10000)
    """

    def __init__(
        self,
        dynamics: "MolecularDynamics",
        property_names: Union[str, List[str]] = "energy",
        sample_interval: int = 1,
        extractors: Optional[Dict[str, Callable[["Atoms"], float]]] = None,
    ):
        """Initialize the sampler.

        Args:
            dynamics: ASE MolecularDynamics object.
            property_names: Names of the properties to monitor. Available options:
                "energy" (or "potential_energy"), "kinetic_energy", "total_energy",
                "volume", "pressure", "temperature", "density".
                Default: "energy".
            sample_interval: Collect property every N steps. Default: 1.
            extractors: Optional dictionary of custom property extractors.
                Keys are property names, values are functions that take an
                Atoms object and return a float.

        Raises:
            CRError: If an invalid property is provided
        """
        self.dynamics = dynamics
        self.property_names = (
            [property_names] if isinstance(property_names, str) else property_names
            )
        cr_check(sample_interval, "sample_interval", int, var_lower_bound=1)
        self.sample_interval = sample_interval
        self._total_steps = 0

        # Set up property extractors
        self.extractors = DEFAULT_EXTRACTORS.copy()
        if extractors:
            self.extractors.update(extractors)

        self._extractors = []
        for property_name in self.property_names:
            # Validate that the requested property has an extractor
            if property_name not in self.extractors:
                available = ", ".join(sorted(self.extractors.keys()))
                raise CRError(
                    f"No extractor available for property: {property_name}. "
                    f"Available properties: {available}"
                )
            self._extractors.append(self.extractors[property_name])

    def __call__(self, nstep: int) -> np.ndarray:
        """Advance the dynamics nstep steps and return property values.

        Uses ASE's observer mechanism for efficient batch execution.

        Note:
            ASE observers fire at step 0 (capturing the initial state) plus
            every ``sample_interval`` steps thereafter. This means requesting
            ``nstep`` samples will return ``nstep + 1`` samples
            (values at steps 0, sample_interval, 2*sample_interval, ...,
            nstep*sample_interval), if the dynamics object has not been
            advanced before this call. Otherwise, ``nstep`` samples
            will be returned.

        Args:
            nstep: Number of samples to collect. The actual number of MD steps
                run will be nstep * sample_interval.

        Returns:
            If self.num_properties == 1, it will return a 1D array of collected
            property values. Otherwise it will return a 2D array of values, with the
            first index being the property. See the note regarding the length
            of the other dimension, which is the number of samples.
        """
        md_steps = nstep * self.sample_interval
        property_values: List[List[float]] = [[] for _ in range(self.num_properties)]

        def collect_properties() -> None:
            """Observer callback to collect property values."""
            for i in range(self.num_properties):
                value = self._extractors[i](self.dynamics.atoms)
                property_values[i].append(value)

        # Attach observer for data collection
        self.dynamics.attach(collect_properties, interval=self.sample_interval)

        try:
            # Run all steps at once (much faster than run(1) in a loop)
            self.dynamics.run(md_steps)
            self._total_steps += md_steps
        finally:
            # Clean up: remove our observer
            for i, (func, *_) in enumerate(self.dynamics.observers):
                if func is collect_properties:
                    self.dynamics.observers.pop(i)
                    break

        return_array = np.asarray(property_values, dtype=np.float64)
        if self.num_properties == 1:
            return_array = np.reshape(return_array, (-1, ))
        return return_array

    @property
    def total_steps(self) -> int:
        """Return the total number of MD steps run so far."""
        return self._total_steps

    @property
    def num_properties(self) -> int:
        """Return the number of properties being sampled"""
        return len(self._extractors)


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

            Note: The ``get_trajectory``, ``fp``, and ``fp_format`` parameters
            are reserved and cannot be overridden.

    Returns:
        Dictionary containing the kim-convergence run_length_control result.
        Key fields include:
        - converged (bool): Whether convergence was achieved.
        - total_run_length (int): Total number of samples collected.
        - equilibration_step (int): Sample at which equilibration was detected.
        - mean (float): Estimated mean of the monitored property.
        - standard_deviation (float): Standard deviation.
        See kim_convergence.run_length_control documentation for full details.

    Raises:
        CRerror:
            If a reserved parameter is provided in ``**kwargs``

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
        >>> sampler = ASESampler(dyn, property_names="temperature")
        >>> result = run_ase_equilibration(
        ...     sampler,
        ...     initial_run_length=1000,
        ...     maximum_run_length=10000,
        ...     relative_accuracy=0.1,
        ... )
        >>> if result["converged"]:
        ...     print(f"Equilibrated! Mean T = {result['mean']:.1f} K")

        With sample_interval to reduce data collection:

        >>> sampler = ASESampler(dyn, property_names="energy", sample_interval=10)
        >>> result = run_ase_equilibration(
        ...     sampler,
        ...     maximum_run_length=5000,  # 5000 samples = 50000 MD steps
        ...     relative_accuracy=0.05,
        ... )

        With multiple properties:

        >>> sampler = ASESampler(dyn, property_names=["energy", "temperature"])

        With custom property extractor:

        >>> def get_max_force(atoms):
        ...     return np.max(np.abs(atoms.get_forces()))
        >>>
        >>> sampler = ASESampler(
        ...     dyn,
        ...     property_names="max_force",
        ...     extractors={"max_force": get_max_force},
        ... )
        >>> result = run_ase_equilibration(sampler, relative_accuracy=0.1)
    """
    # Build kwargs for run_length_control
    # Prevent override of reserved parameters that are critical to function operation
    reserved_keys = {"get_trajectory", "fp", "fp_format", "number_of_variables"}
    conflicts = reserved_keys.intersection(kwargs)
    if conflicts:
        raise CRError(
            f"Cannot override reserved parameter(s): {', '.join(sorted(conflicts))}. "
            "'get_trajectory' and 'number_of_variables' are automatically set from the "
            "provided sampler, and 'fp'/'fp_format' are required for result parsing."
        )

    rlc_kwargs: Dict[str, Any] = {
        "get_trajectory": sampler,
        "number_of_variables": sampler.num_properties,
        "fp": "return",
        "fp_format": "json",
        **kwargs,
    }

    # Run convergence control
    result_json = run_length_control(**rlc_kwargs)

    return json.loads(result_json)
