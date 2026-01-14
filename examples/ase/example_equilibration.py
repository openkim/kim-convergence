"""Example: Convergence-controlled equilibration with ASE.

This example demonstrates how to use kim-convergence to automatically
detect equilibration in ASE molecular dynamics simulations.
"""

import numpy as np
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from kim_convergence.ase import ASESampler, run_ase_equilibration


def main():
    """Run a simple equilibration example with copper."""
    # Create a copper bulk system
    atoms = bulk("Cu", cubic=True) * (4, 4, 4)  # 256 atoms
    atoms.calc = EMT()

    # Randomize velocities for initial temperature
    MaxwellBoltzmannDistribution(atoms, temperature_K=600)

    # Create Langevin dynamics
    dyn = Langevin(
        atoms,
        timestep=5 * units.fs,
        temperature_K=500,
        friction=0.01 / units.fs,
    )

    print("Starting convergence-controlled equilibration...")
    print(f"Target temperature: 500 K")
    print(f"Number of atoms: {len(atoms)}")
    print()

    # Create sampler and run equilibration
    sampler = ASESampler(dyn, property_name="temperature")
    result = run_ase_equilibration(
        sampler,
        initial_run_length=500,
        maximum_run_length=20000,
        relative_accuracy=0.1,  # 10% relative accuracy
        confidence_coefficient=0.95,
    )

    # Print results
    print("\n" + "=" * 50)
    print("EQUILIBRATION RESULTS")
    print("=" * 50)

    if result["converged"]:
        print(f"✓ Converged after {result['total_run_length']} samples")
        print(f"  Equilibration detected at sample: {result['equilibration_step']}")
        print(f"  Mean temperature: {result.get('mean', 'N/A')}")
    else:
        print(f"✗ Did not converge within {result['total_run_length']} samples")
        print("  Consider increasing maximum_run_length")

    return result


def example_with_sample_interval():
    """Example showing sample_interval for expensive calculators.

    When using expensive calculators (e.g., neural network potentials),
    you may want to sample less frequently to reduce overhead.
    """
    atoms = bulk("Cu", cubic=True) * (3, 3, 3)
    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(atoms, temperature_K=600)

    dyn = Langevin(
        atoms,
        timestep=5 * units.fs,
        temperature_K=500,
        friction=0.01 / units.fs,
    )

    print("\nEquilibrating with sample_interval=10 (sample every 10 MD steps)...")

    # Sample every 10 MD steps
    sampler = ASESampler(dyn, property_name="temperature", sample_interval=10)
    result = run_ase_equilibration(
        sampler,
        initial_run_length=100,  # 100 samples = 1000 MD steps
        maximum_run_length=2000,  # 2000 samples = 20000 MD steps
        relative_accuracy=0.1,
    )

    print(f"Converged: {result['converged']}")
    print(f"Total samples: {result['total_run_length']}")
    print(f"Total MD steps: {sampler.total_steps}")

    return result


def example_with_custom_extractor():
    """Example showing how to use a custom property extractor."""
    atoms = bulk("Cu", cubic=True) * (3, 3, 3)
    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(atoms, temperature_K=500)

    dyn = Langevin(
        atoms,
        timestep=5 * units.fs,
        temperature_K=400,
        friction=0.01 / units.fs,
    )

    # Define a custom extractor for maximum force component
    def get_max_force(atoms):
        """Get the maximum absolute force component."""
        return float(np.max(np.abs(atoms.get_forces())))

    print("\nEquilibrating with custom extractor (max force)...")

    sampler = ASESampler(
        dyn,
        property_name="max_force",
        extractors={"max_force": get_max_force},
    )
    result = run_ase_equilibration(
        sampler,
        initial_run_length=200,
        maximum_run_length=5000,
        relative_accuracy=0.2,
    )

    print(f"Converged: {result['converged']}")

    return result


def example_energy_equilibration():
    """Example monitoring potential energy."""
    atoms = bulk("Cu", cubic=True) * (3, 3, 3)
    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(atoms, temperature_K=800)

    dyn = Langevin(
        atoms,
        timestep=5 * units.fs,
        temperature_K=500,
        friction=0.02 / units.fs,
    )

    print("\nEquilibrating based on potential energy...")

    sampler = ASESampler(dyn, property_name="energy")
    result = run_ase_equilibration(
        sampler,
        initial_run_length=500,
        maximum_run_length=15000,
        relative_accuracy=0.01,  # 1% relative accuracy (stricter)
    )

    print(f"Converged: {result['converged']}")

    return result


if __name__ == "__main__":
    main()
    print("\n" + "-" * 50 + "\n")
    example_with_sample_interval()
    print("\n" + "-" * 50 + "\n")
    example_energy_equilibration()
    print("\n" + "-" * 50 + "\n")
    example_with_custom_extractor()
