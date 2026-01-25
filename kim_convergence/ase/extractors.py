"""Property extractors for ASE Atoms objects.

This module provides functions to extract thermodynamic properties from
ASE Atoms objects for use in convergence analysis.
"""

from typing import TYPE_CHECKING, Callable, Dict

if TYPE_CHECKING:
    from ase.atoms import Atoms


def get_potential_energy(atoms: "Atoms") -> float:
    """Extract potential energy from atoms object.

    Args:
        atoms: ASE Atoms object with an attached calculator.

    Returns:
        Potential energy in eV.
    """
    return float(atoms.get_potential_energy())


def get_kinetic_energy(atoms: "Atoms") -> float:
    """Extract kinetic energy from atoms object.

    Args:
        atoms: ASE Atoms object.

    Returns:
        Kinetic energy in eV.
    """
    return float(atoms.get_kinetic_energy())


def get_total_energy(atoms: "Atoms") -> float:
    """Extract total energy (potential + kinetic) from atoms object.

    Args:
        atoms: ASE Atoms object with an attached calculator.

    Returns:
        Total energy in eV.
    """
    return float(atoms.get_potential_energy() + atoms.get_kinetic_energy())


def get_volume(atoms: "Atoms") -> float:
    """Extract volume from atoms object.

    Args:
        atoms: ASE Atoms object.

    Returns:
        Volume in Å³.
    """
    return float(atoms.get_volume())


def get_pressure(atoms: "Atoms") -> float:
    """Extract pressure from atoms object.

    The pressure is computed as the negative trace of the stress tensor
    divided by 3 (hydrostatic pressure).

    Args:
        atoms: ASE Atoms object with an attached calculator that supports
            stress calculations.

    Returns:
        Pressure in eV/Å³.
    """
    import numpy as np

    stress = atoms.get_stress(voigt=False)  # 3x3 stress tensor
    return float(-np.trace(stress) / 3.0)


def get_temperature(atoms: "Atoms") -> float:
    """Extract kinetic temperature from atoms object.

    Args:
        atoms: ASE Atoms object with velocities.

    Returns:
        Temperature in Kelvin.
    """
    return float(atoms.get_temperature())


def get_density(atoms: "Atoms") -> float:
    """Extract density from atoms object.

    Args:
        atoms: ASE Atoms object.

    Returns:
        Density in g/cm³.
    """
    # Conversion factors:
    # 1 amu = 1.66053906660e-24 g (CODATA 2018)
    # 1 Å³ = 1e-24 cm³
    AMU_TO_GRAMS = 1.66053906660e-24

    mass_amu = sum(atoms.get_masses())  # Total mass in amu
    volume_ang3 = atoms.get_volume()  # Volume in Å³

    mass_g = mass_amu * AMU_TO_GRAMS
    volume_cm3 = volume_ang3 * 1e-24

    return float(mass_g / volume_cm3)


# Default property extractors mapping
DEFAULT_EXTRACTORS: Dict[str, Callable[["Atoms"], float]] = {
    "potential_energy": get_potential_energy,
    "kinetic_energy": get_kinetic_energy,
    "total_energy": get_total_energy,
    "energy": get_potential_energy,  # Alias for potential_energy
    "volume": get_volume,
    "pressure": get_pressure,
    "temperature": get_temperature,
    "density": get_density,
}
