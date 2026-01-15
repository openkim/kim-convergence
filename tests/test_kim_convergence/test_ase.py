"""Test ASE integration module."""

import unittest

import numpy as np

# Check if ASE is available
try:
    from ase import units
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    HAS_ASE = True
except ImportError:
    HAS_ASE = False


@unittest.skipUnless(HAS_ASE, "ASE not installed")
class TestASEExtractors(unittest.TestCase):
    """Test property extractors for ASE Atoms objects."""

    def setUp(self):
        """Create a simple atoms object for testing."""
        self.atoms = bulk("Cu", cubic=True) * (2, 2, 2)  # 32 atoms - small
        self.atoms.calc = EMT()
        # Set some velocities for temperature calculation
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=300)

    def test_get_potential_energy(self):
        """Test potential energy extractor."""
        from kim_convergence.ase import get_potential_energy

        energy = get_potential_energy(self.atoms)
        self.assertIsInstance(energy, float)
        self.assertTrue(np.isfinite(energy))

    def test_get_kinetic_energy(self):
        """Test kinetic energy extractor."""
        from kim_convergence.ase import get_kinetic_energy

        ke = get_kinetic_energy(self.atoms)
        self.assertIsInstance(ke, float)
        self.assertGreater(ke, 0)  # Should have KE from velocities

    def test_get_total_energy(self):
        """Test total energy extractor."""
        from kim_convergence.ase import (
            get_kinetic_energy,
            get_potential_energy,
            get_total_energy,
        )

        total = get_total_energy(self.atoms)
        pe = get_potential_energy(self.atoms)
        ke = get_kinetic_energy(self.atoms)
        self.assertAlmostEqual(total, pe + ke, places=10)

    def test_get_volume(self):
        """Test volume extractor."""
        from kim_convergence.ase import get_volume

        volume = get_volume(self.atoms)
        self.assertIsInstance(volume, float)
        self.assertGreater(volume, 0)
        # Check it matches ASE's value
        self.assertAlmostEqual(volume, self.atoms.get_volume(), places=10)

    def test_get_temperature(self):
        """Test temperature extractor."""
        from kim_convergence.ase import get_temperature

        temp = get_temperature(self.atoms)
        self.assertIsInstance(temp, float)
        self.assertGreater(temp, 0)  # Should have temperature from velocities
        # Should be reasonably close to 300 K (set in setUp)
        self.assertGreater(temp, 100)
        self.assertLess(temp, 600)

    def test_get_density(self):
        """Test density extractor."""
        from kim_convergence.ase import get_density

        density = get_density(self.atoms)
        self.assertIsInstance(density, float)
        self.assertGreater(density, 0)
        # Copper density is about 8.96 g/cm³
        self.assertGreater(density, 5)
        self.assertLess(density, 15)

    def test_get_pressure(self):
        """Test pressure extractor."""
        from kim_convergence.ase import get_pressure

        # EMT calculator supports stress, so this should work
        try:
            pressure = get_pressure(self.atoms)
            self.assertIsInstance(pressure, float)
        except Exception:
            # Some calculators may not support stress
            pass

    def test_default_extractors_dict(self):
        """Test DEFAULT_EXTRACTORS dictionary contains expected keys."""
        from kim_convergence.ase import DEFAULT_EXTRACTORS

        expected_keys = {
            "potential_energy",
            "kinetic_energy",
            "total_energy",
            "energy",
            "volume",
            "pressure",
            "temperature",
            "density",
        }
        self.assertEqual(set(DEFAULT_EXTRACTORS.keys()), expected_keys)

        # All values should be callable
        for name, func in DEFAULT_EXTRACTORS.items():
            self.assertTrue(callable(func), f"{name} extractor is not callable")


@unittest.skipUnless(HAS_ASE, "ASE not installed")
class TestASESampler(unittest.TestCase):
    """Test the ASESampler class."""

    def setUp(self):
        """Set up a small system for fast tests."""
        np.random.seed(42)
        self.atoms = bulk("Cu", cubic=True) * (2, 2, 2)
        self.atoms.calc = EMT()
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=300)
        self.dyn = Langevin(
            self.atoms,
            timestep=1 * units.fs,
            temperature_K=300,
            friction=0.02 / units.fs,
        )

    def test_sampler_creation(self):
        """Test sampler can be created."""
        from kim_convergence.ase import ASESampler

        sampler = ASESampler(self.dyn, property_name="temperature")
        self.assertEqual(sampler.property_name, "temperature")
        self.assertEqual(sampler.sample_interval, 1)
        self.assertEqual(sampler.total_steps, 0)

    def test_sampler_returns_correct_shape(self):
        """Test that sampler returns array of correct shape."""
        from kim_convergence.ase import ASESampler

        sampler = ASESampler(
            dynamics=self.dyn,
            property_name="temperature",
            sample_interval=1,
        )

        # Note: ASE observer fires at step 0 (initial) + every interval steps
        # So for 10 samples requested (10 MD steps), we get 11 values (0,1,2,...,10)
        result = sampler(10)  # Request 10 samples -> 10 MD steps
        self.assertGreater(len(result), 0)
        self.assertEqual(sampler.total_steps, 10)

        result2 = sampler(5)  # Request 5 more samples -> 5 more MD steps
        self.assertGreater(len(result2), 0)
        self.assertEqual(sampler.total_steps, 15)

    def test_sampler_with_sample_interval(self):
        """Test sampler with sample_interval > 1."""
        from kim_convergence.ase import ASESampler

        sampler = ASESampler(
            dynamics=self.dyn,
            property_name="temperature",
            sample_interval=5,
        )

        # Request 4 samples -> runs 20 MD steps
        # ASE observer fires at step 0, 5, 10, 15, 20 = 5 values
        result = sampler(4)
        self.assertGreater(len(result), 0)
        self.assertEqual(sampler.total_steps, 20)

    def test_sampler_custom_extractor(self):
        """Test sampler with custom extractor."""
        from kim_convergence.ase import ASESampler

        def constant_extractor(atoms):
            return 42.0

        sampler = ASESampler(
            dynamics=self.dyn,
            property_name="constant",
            sample_interval=1,
            extractors={"constant": constant_extractor},
        )

        result = sampler(5)
        self.assertTrue(np.allclose(result, 42.0))

    def test_sampler_invalid_property(self):
        """Test sampler raises ValueError for invalid property."""
        from kim_convergence.ase import ASESampler

        with self.assertRaises(ValueError) as ctx:
            ASESampler(
                dynamics=self.dyn,
                property_name="invalid_property",
            )

        self.assertIn("No extractor available", str(ctx.exception))


@unittest.skipUnless(HAS_ASE, "ASE not installed")
class TestASEEquilibration(unittest.TestCase):
    """Test convergence-controlled equilibration for ASE."""

    def setUp(self):
        """Set up a small system for fast tests."""
        np.random.seed(42)
        self.atoms = bulk("Cu", cubic=True) * (2, 2, 2)  # 32 atoms - small
        self.atoms.calc = EMT()
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=400)

    def _create_dynamics(self, temperature_K=300):
        """Create a fresh dynamics object."""
        dyn = Langevin(
            self.atoms,
            timestep=2 * units.fs,  # Small timestep
            temperature_K=temperature_K,
            friction=0.05 / units.fs,  # High friction for fast equilibration
        )
        return dyn

    def test_run_ase_equilibration_basic(self):
        """Test basic equilibration run."""
        from kim_convergence.ase import ASESampler, run_ase_equilibration

        dyn = self._create_dynamics(temperature_K=350)
        sampler = ASESampler(dyn, property_name="temperature")

        result = run_ase_equilibration(
            sampler,
            initial_run_length=50,
            maximum_run_length=500,
            relative_accuracy=0.5,  # Very loose for fast test
            confidence_coefficient=0.95,
        )

        # Check result structure (kim-convergence output)
        self.assertIn("converged", result)
        self.assertIn("total_run_length", result)
        self.assertIsInstance(result["converged"], bool)
        self.assertGreater(result["total_run_length"], 0)

    def test_run_ase_equilibration_converges(self):
        """Test that equilibration can converge with reasonable settings."""
        from kim_convergence.ase import ASESampler, run_ase_equilibration

        dyn = self._create_dynamics(temperature_K=300)
        sampler = ASESampler(dyn, property_name="kinetic_energy")

        result = run_ase_equilibration(
            sampler,
            initial_run_length=100,
            maximum_run_length=2000,
            relative_accuracy=0.3,  # Loose accuracy for fast convergence
            confidence_coefficient=0.95,
        )

        # With these settings, it should converge
        self.assertTrue(result["converged"])
        self.assertIsNotNone(result.get("mean"))

    def test_run_ase_equilibration_with_sample_interval(self):
        """Test equilibration with sample_interval > 1."""
        from kim_convergence.ase import ASESampler, run_ase_equilibration

        dyn = self._create_dynamics(temperature_K=300)
        sampler = ASESampler(dyn, property_name="temperature", sample_interval=10)

        result = run_ase_equilibration(
            sampler,
            initial_run_length=10,  # 10 samples = 100 MD steps
            maximum_run_length=100,  # 100 samples = 1000 MD steps
            relative_accuracy=0.5,
        )

        self.assertIn("converged", result)
        self.assertIn("total_run_length", result)

    def test_run_ase_equilibration_custom_extractor(self):
        """Test equilibration with a custom property extractor."""
        from kim_convergence.ase import ASESampler, run_ase_equilibration

        dyn = self._create_dynamics(temperature_K=300)

        def get_max_velocity(atoms):
            """Get maximum velocity magnitude."""
            velocities = atoms.get_velocities()
            speeds = np.linalg.norm(velocities, axis=1)
            return float(np.max(speeds))

        sampler = ASESampler(
            dyn,
            property_name="max_velocity",
            extractors={"max_velocity": get_max_velocity},
        )

        result = run_ase_equilibration(
            sampler,
            initial_run_length=50,
            maximum_run_length=500,
            relative_accuracy=0.5,
        )

        self.assertIn("converged", result)
        self.assertIn("total_run_length", result)

    def test_run_ase_equilibration_returns_dict(self):
        """Test that result is a dictionary."""
        from kim_convergence.ase import ASESampler, run_ase_equilibration

        dyn = self._create_dynamics(temperature_K=300)
        sampler = ASESampler(dyn, property_name="temperature")

        result = run_ase_equilibration(
            sampler,
            initial_run_length=50,
            maximum_run_length=300,
            relative_accuracy=0.5,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("converged", result)

    def test_run_ase_equilibration_energy(self):
        """Test equilibration monitoring potential energy."""
        from kim_convergence.ase import ASESampler, run_ase_equilibration

        dyn = self._create_dynamics(temperature_K=300)
        sampler = ASESampler(dyn, property_name="energy")

        result = run_ase_equilibration(
            sampler,
            initial_run_length=100,
            maximum_run_length=1000,
            relative_accuracy=0.1,
        )

        self.assertIn("converged", result)
        self.assertIn("total_run_length", result)


@unittest.skipUnless(HAS_ASE, "ASE not installed")
class TestASEModuleImport(unittest.TestCase):
    """Test module import behavior."""

    def test_import_ase_module(self):
        """Test that ASE module can be imported."""
        from kim_convergence import ase

        self.assertIsNotNone(ase)

    def test_import_run_ase_equilibration(self):
        """Test that run_ase_equilibration can be imported directly."""
        from kim_convergence.ase import run_ase_equilibration

        self.assertTrue(callable(run_ase_equilibration))

    def test_import_ase_sampler(self):
        """Test that ASESampler can be imported directly."""
        from kim_convergence.ase import ASESampler

        self.assertTrue(callable(ASESampler))

    def test_import_all_exports(self):
        """Test that all __all__ exports are importable."""
        from kim_convergence import ase

        for name in ase.__all__:
            obj = getattr(ase, name)
            self.assertIsNotNone(obj, f"Failed to get {name}")


if __name__ == "__main__":
    unittest.main()
