"""Test run_length_control module."""

import json
import kim_edn
import numpy as np
import os
import tempfile
from typing import cast
import unittest

try:
    import kim_convergence as cr
except Exception as e:  # intentional catch-all
    raise RuntimeError("Failed to import `kim-convergence` utility module") from e

from kim_convergence import CRError


class TestRunLengthControl(unittest.TestCase):
    """Test run_length_control module components."""

    def setUp(self):
        """Setup random state for reproducibility."""
        self.rng = np.random.RandomState(12345)

    def test_repr_smoke(self):
        """The public function must be importable and have a sane representation."""

        func = cr.run_length_control
        self.assertTrue(callable(func))
        self.assertEqual(func.__name__, "run_length_control")
        self.assertIn("run_length_control", repr(func))
        self.assertIn("run_length_control", str(func))

    def test_invalid_inputs(self):
        """All illegal argument combinations must raise ``CRError`` immediately."""

        # a constant trajectory
        def constant_trajectory(step) -> np.ndarray:
            return np.full(step, 5.0)

        # Base kwargs to suppress warning and keep run short/fast
        base_kwargs = {
            "get_trajectory": constant_trajectory,
            "number_of_variables": 1,
            "initial_run_length": 10,
            "run_length_factor": 1.5,
            "maximum_run_length": 100,
            "maximum_equilibration_step": 50,
            "relative_accuracy": 0.01,
            "confidence_coefficient": 0.95,
            "confidence_interval_approximation_method": "uncorrelated_sample",
        }

        # 1. get_trajectory
        self.assertRaises(CRError, cr.run_length_control, None)
        self.assertRaises(CRError, cr.run_length_control, "not callable")
        self.assertRaises(
            CRError, cr.run_length_control, lambda x: x
        )  # wrong signature

        # 2. Scalar parameters
        for key, bad_value in [
            ("number_of_variables", 0),
            ("number_of_variables", -5),
            ("number_of_variables", 1.5),
            ("initial_run_length", 0),
            ("initial_run_length", -10),
            ("run_length_factor", -0.01),
            ("run_length_factor", -1.0),
            ("run_length_factor", "invalid"),
            ("maximum_run_length", 0),
            ("maximum_run_length", -100),
            ("maximum_equilibration_step", 0),
            ("maximum_equilibration_step", 101),
            ("maximum_equilibration_step", -5),
            ("minimum_number_of_independent_samples", 0),
            ("minimum_number_of_independent_samples", -10),
            ("confidence_coefficient", -0.1),
            ("confidence_coefficient", 1.1),
            ("confidence_coefficient", "invalid"),
            ("heidel_welch_number_points", 0),
            ("heidel_welch_number_points", -5),
            ("heidel_welch_number_points", "invalid"),
            ("confidence_interval_approximation_method", "non_existent_method"),
            ("number_of_cores", 0),
            ("number_of_cores", 1.0),
            ("number_of_cores", "invalid"),
            ("minimum_correlation_time", 0),
            ("minimum_correlation_time", 1.0),
            ("minimum_correlation_time", "invalid"),
            ("fp", "some_file.txt"),  # a string but not "return"
            ("fp", 12345),  # no write()
            ("fp_format", "xml"),
        ]:
            kwargs = base_kwargs.copy()
            kwargs[key] = bad_value
            self.assertRaises(CRError, cr.run_length_control, **kwargs)

        # 3. Accuracy specification
        # Both None
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "relative_accuracy": None,
                "absolute_accuracy": None,
            },
        )

        # Negative values
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "relative_accuracy": -0.01,
            },
        )
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "absolute_accuracy": -0.01,
            },
        )

        # list length mismatch
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "number_of_variables": 2,
                "relative_accuracy": [0.01],  # Too short
            },
        )
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "number_of_variables": 2,
                "absolute_accuracy": [0.01, 0.01, 0.01],  # Too long
            },
        )

        # 4. Population parameter conflicts
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "population_cdf": "norm",  # Custom dist
                "population_mean": 0.0,  # Not allowed with cdf
            },
        )
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "population_cdf": "norm",
                "population_standard_deviation": 1.0,
            },
        )
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "population_cdf": None,  # Normal assumed
                "population_args": (1,),  # Not allowed without cdf
            },
        )
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "population_cdf": None,
                "population_loc": 0.0,
            },
        )
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "population_cdf": None,
                "population_scale": 1.0,
            },
        )
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "population_cdf": None,
                "population_standard_deviation": -1.0,  # Must be >0
            },
        )

        # Invalid custom CDF name
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "population_cdf": "invalid_distribution_name",
            },
        )

        # 5. Invalid UCL method
        self.assertRaises(
            CRError,
            cr.run_length_control,
            **{
                **base_kwargs,
                "confidence_interval_approximation_method": "non_existent_method",
            },
        )

        # 6. Non-finite data from trajectory
        def nan_trajectory(step) -> np.ndarray:
            return np.full(step, np.nan)

        def inf_trajectory(step) -> np.ndarray:
            return np.full(step, np.inf)

        self.assertRaises(
            CRError,
            cr.run_length_control,
            nan_trajectory,
            **{k: v for k, v in base_kwargs.items() if k != "get_trajectory"},
        )
        self.assertRaises(
            CRError,
            cr.run_length_control,
            inf_trajectory,
            **{k: v for k, v in base_kwargs.items() if k != "get_trajectory"},
        )

        # 7. Wrong trajectory shape/dimension
        def wrong_shape_trajectory(step) -> np.ndarray:
            return np.ones((2, step))  # 2 vars but number_of_variables=1

        self.assertRaises(
            CRError,
            cr.run_length_control,
            wrong_shape_trajectory,
            **{k: v for k, v in base_kwargs.items() if k != "get_trajectory"},
        )

        def wrong_ndim_trajectory(step) -> np.ndarray:
            return np.ones((step, step))  # 2D when 1D expected

        self.assertRaises(
            CRError,
            cr.run_length_control,
            wrong_ndim_trajectory,
            **{k: v for k, v in base_kwargs.items() if k != "get_trajectory"},
        )

    def test_constant_data_single_variable(self):
        """A perfectly constant series must converge instantly (UCL = 0)."""

        # a constant trajectory
        def constant_trajectory(step) -> np.ndarray:
            return np.full(step, 5.0)

        msg = cr.run_length_control(
            get_trajectory=constant_trajectory,
            number_of_variables=1,
            initial_run_length=10,
            run_length_factor=1.5,
            maximum_run_length=100,
            relative_accuracy=0.01,
            absolute_accuracy=None,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method="uncorrelated_sample",
            fp="return",
            fp_format="json",
        )

        self.assertIsInstance(msg, str)
        assert isinstance(msg, str)  # keeps mypy happy
        json_obj = json.loads(msg)

        self.assertTrue(json_obj["converged"])
        self.assertTrue(json_obj["equilibration_detected"])
        self.assertEqual(json_obj["equilibration_step"], 0)  # No warm-up needed
        self.assertAlmostEqual(json_obj["mean"], 5.0)
        self.assertAlmostEqual(json_obj["standard_deviation"], 0)
        self.assertEqual(json_obj["effective_sample_size"], 1)

    def test_absolute_accuracy_only(self):
        """Run with *only* absolute accuracy (relative_accuracy = None)."""

        # a constant trajectory
        def constant_trajectory(step) -> np.ndarray:
            return np.full(step, 10.0)

        msg = cr.run_length_control(
            get_trajectory=constant_trajectory,
            number_of_variables=1,
            initial_run_length=10,
            maximum_run_length=100,
            relative_accuracy=None,
            absolute_accuracy=0.5,
            fp="return",
            fp_format="json",
        )
        self.assertIsInstance(msg, str)
        assert isinstance(msg, str)  # keeps mypy happy
        obj = json.loads(msg)

        self.assertTrue(obj["converged"])
        self.assertEqual(obj["relative_accuracy"], "None")
        self.assertEqual(obj["absolute_accuracy"], 0.5)
        # verify UCL is effectively zero
        self.assertAlmostEqual(obj["upper_confidence_limit"], 0.0)
        self.assertAlmostEqual(obj["mean"], 10.0)

    def test_max_run_length_hit(self):
        """A noisy series that never reaches the requested accuracy must stop at ``maximum_run_length``."""

        # trajectory
        def slow_trajectory(step) -> np.ndarray:
            # large amplitude -> relative half-width stays big
            return 1.0 + 0.3 * self.rng.randn(step)  # std ≈ 0.3

        converged = cr.run_length_control(
            get_trajectory=slow_trajectory,
            number_of_variables=1,
            initial_run_length=50,
            run_length_factor=1.1,
            maximum_run_length=200,  # hard cap
            relative_accuracy=0.001,  # unattainable
            fp=None,  # stdout suppressed
        )
        self.assertFalse(converged)

    def test_max_equilibration_hit(self):
        """A strong linear drift forces MSER to keep moving the truncation point forward."""

        # a drifting trajectory
        def drifting_trajectory(step) -> np.ndarray:
            # linear drift -> MSER will keep pushing truncation forward
            return np.arange(step) * 0.5

        with self.assertRaises(cr.CRError) as ctx:
            cr.run_length_control(
                get_trajectory=drifting_trajectory,
                number_of_variables=1,
                initial_run_length=150,
                maximum_run_length=300,
                maximum_equilibration_step=60,  # lower than possible detection
                relative_accuracy=0.001,
                dump_trajectory=False,
            )
        self.assertIn("equilibration", str(ctx.exception).lower())

    def test_dump_and_edn_on_failure(self):
        """When convergence fails the full trajectory must be written to the requested file."""

        with tempfile.NamedTemporaryFile(suffix=".edn", delete=False) as f:
            tmp = f.name

        # a constant trajectory
        def constant_trajectory(step) -> np.ndarray:
            return np.full(step, 42.0)

        try:
            msg = cr.run_length_control(
                get_trajectory=constant_trajectory,
                number_of_variables=1,
                initial_run_length=100,
                maximum_run_length=500,
                relative_accuracy=0.05,
                dump_trajectory=True,
                dump_trajectory_fp=tmp,
                fp="return",
                fp_format="edn",
            )
            self.assertIsInstance(msg, str)
            obj: dict = cast(dict, kim_edn.loads(msg))

            self.assertTrue(obj["converged"])
            self.assertEqual(obj["mean"], 42.0)
            self.assertEqual(obj["upper_confidence_limit"], 0.0)
            self.assertAlmostEqual(obj["effective_sample_size"], 1.0)
            self.assertIn("requested_sample_size", obj)

            # file must contain the trajectory
            with open(tmp) as f:
                trajectory = kim_edn.load(f)

            traj_array = np.array(trajectory)
            self.assertEqual(traj_array.shape, (100,))  # initial_run_length
            self.assertTrue(np.allclose(traj_array, 42.0))

        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_population_cdf_levene(self):
        """Gamma-distributed data with known parameters must pass the Levene test."""

        # a trajectory
        def gamma_trajectory(step) -> np.ndarray:
            # gamma(2, scale=2) has mean=4, var=8
            return self.rng.gamma(shape=2, scale=2, size=step)

        converged = cr.run_length_control(
            get_trajectory=gamma_trajectory,
            number_of_variables=1,
            initial_run_length=500,
            maximum_run_length=2000,
            relative_accuracy=0.1,
            population_cdf="gamma",
            population_args=(2,),
            population_scale=2.0,
            confidence_coefficient=0.95,
            fp=None,
        )
        # Levene test must pass -> convergence achieved
        self.assertTrue(converged)

    def test_relative_accuracy_convergence(self):
        """Test successful convergence using relative accuracy with noisy data."""

        # a noisy trajectory
        def noisy_trajectory(step) -> np.ndarray:
            # Mean ≈ 10.0, std ≈ 1.0 -> relative half-width decreases with more data
            return 10.0 + self.rng.normal(0, 1, step)

        msg = cr.run_length_control(
            get_trajectory=noisy_trajectory,
            number_of_variables=1,
            initial_run_length=500,
            run_length_factor=1.5,
            maximum_run_length=100_000,
            relative_accuracy=0.05,  # 5% relative accuracy
            absolute_accuracy=None,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method="uncorrelated_sample",
            fp="return",
            fp_format="json",
        )
        self.assertIsInstance(msg, str)
        assert isinstance(msg, str)  # keeps mypy happy
        obj = json.loads(msg)

        self.assertTrue(obj["converged"])
        self.assertTrue(obj["equilibration_detected"])
        self.assertAlmostEqual(obj["mean"], 10.0, delta=0.2)
        self.assertAlmostEqual(obj["relative_half_width"], 0.02, delta=0.01)
        self.assertGreater(obj["effective_sample_size"], 100)
        self.assertIn("confidence_interval", obj)
        # confirm UCL is roughly half-width
        self.assertAlmostEqual(
            obj["upper_confidence_limit"],
            obj["relative_half_width"] * abs(obj["mean"]),
            delta=1e-6,
        )

    def test_trajectory_with_args(self):
        """Test the alternative get_trajectory signature that accepts extra args."""
        trajectory_state = {"current": 0, "values": []}

        # a trajectory with args
        def with_args_trajectory(step: int, args: dict) -> np.ndarray:
            start = args["current"]
            end = start + step
            data = np.full(step, 7.5)  # constant for fast equilibration/convergence
            args["current"] = end
            args["values"].extend(data.tolist())
            return data

        msg = cr.run_length_control(
            get_trajectory=with_args_trajectory,
            get_trajectory_args=trajectory_state,
            number_of_variables=1,
            initial_run_length=200,
            maximum_run_length=10_000,
            relative_accuracy=0.01,
            fp="return",
            fp_format="edn",
        )
        self.assertIsInstance(msg, str)
        obj: dict = cast(dict, kim_edn.loads(msg))

        self.assertTrue(obj["converged"])
        self.assertAlmostEqual(obj["mean"], 7.5, delta=0.1)
        # Verify that the state was mutated correctly
        total_steps = trajectory_state["current"]
        self.assertGreaterEqual(total_steps, obj["total_run_length"])
        self.assertAlmostEqual(np.mean(trajectory_state["values"]), 7.5, delta=1e-6)

    def test_minimum_independent_samples_enforced(self):
        """Test that convergence requires minimum_number_of_independent_samples."""

        # a constant trajectory
        def constant_trajectory(step) -> np.ndarray:
            return np.full(step, 3.14)  # perfectly constant -> ess ≈ 1

        # With a high minimum, it should hit max_run_length and NOT converge
        converged = cr.run_length_control(
            get_trajectory=constant_trajectory,
            number_of_variables=1,
            initial_run_length=1000,
            maximum_run_length=5000,
            relative_accuracy=0.01,
            minimum_number_of_independent_samples=100,  # impossible with constant data
            fp=None,  # suppress output
        )
        self.assertFalse(converged)

    def test_minimum_independent_samples_met(self):
        """Test convergence when minimum independent samples is achievable."""

        # a uncorrelated trajectory
        def uncorrelated_trajectory(step) -> np.ndarray:
            # Uncorrelated noise -> effective sample size ≈ total steps
            return self.rng.normal(0, 1, step)

        # with self.assertWarnsRegex(UserWarning, "confidence interval includes zero"):
        msg = cr.run_length_control(
            get_trajectory=uncorrelated_trajectory,
            number_of_variables=1,
            initial_run_length=5000,
            maximum_run_length=100_000,
            relative_accuracy=0.1,
            minimum_number_of_independent_samples=2000,
            confidence_interval_approximation_method="uncorrelated_sample",
            fp="return",
            fp_format="json",
        )
        self.assertIsInstance(msg, str)
        assert isinstance(msg, str)  # keeps mypy happy
        obj = json.loads(msg)

        self.assertFalse(obj["converged"])
        self.assertEqual(obj["total_run_length"], 100_000)
        self.assertGreaterEqual(obj["effective_sample_size"], 2000)
        self.assertAlmostEqual(obj["mean"], 0.0, delta=0.2)
        self.assertTrue(obj["relative_accuracy_undefined"])

    def test_multi_variable_mixed_accuracy_constant_data(self):
        """Test multi-variable constant data with mixed relative/absolute accuracy specifications."""

        # a multi-variable constant trajectory
        def multivar_constant_trajectory(step) -> np.ndarray:
            return np.full((2, step), 5.0)  # 2 variables, each = 5.0

        msg = cr.run_length_control(
            get_trajectory=multivar_constant_trajectory,
            number_of_variables=2,
            initial_run_length=20,
            run_length_factor=1.2,
            maximum_run_length=200,
            relative_accuracy=[0.05, None],  # var-0 relative, var-1 absolute
            absolute_accuracy=[0.1, 0.02],
            population_mean=[None, None],
            population_standard_deviation=[None, None],
            confidence_coefficient=0.95,
            confidence_interval_approximation_method="uncorrelated_sample",
            fp="return",
            fp_format="edn",
        )
        self.assertIsInstance(msg, str)
        obj: dict = cast(dict, kim_edn.loads(msg))

        self.assertTrue(obj["converged"])
        self.assertTrue(obj["equilibration_detected"])

        # EDN nests per-variable dicts under "0", "1"
        self.assertAlmostEqual(obj["0"]["mean"], 5.0)
        self.assertAlmostEqual(obj["1"]["mean"], 5.0)
        self.assertEqual(obj["0"]["relative_accuracy"], 0.05)
        self.assertEqual(obj["1"]["relative_accuracy"], "None")
        self.assertEqual(obj["1"]["absolute_accuracy"], 0.02)

    def test_multi_variable_different_accuracies_noisy(self):
        """Test convergence when variables have different accuracy requirements and one is noisy."""

        # a multi-variable constant trajectory
        def multivar_constant_trajectory(step) -> np.ndarray:
            var0 = np.full(step, 5.0)
            var1 = 10.0 + self.rng.normal(0, 1, step)
            return np.vstack([var0, var1])

        msg = cr.run_length_control(
            get_trajectory=multivar_constant_trajectory,
            number_of_variables=2,
            initial_run_length=2000,
            run_length_factor=1.5,
            maximum_run_length=300_000,
            relative_accuracy=[0.01, 0.05],
            absolute_accuracy=None,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method="uncorrelated_sample",
            fp="return",
            fp_format="json",
        )
        self.assertIsInstance(msg, str)
        assert isinstance(msg, str)  # keeps mypy happy
        obj = json.loads(msg)

        self.assertTrue(obj["converged"])
        self.assertTrue(obj["equilibration_detected"])

        self.assertIn("0", obj)
        self.assertIn("1", obj)

        var0 = obj["0"]
        var1 = obj["1"]

        self.assertAlmostEqual(var0["mean"], 5.0)
        self.assertLess(var0["relative_half_width"], 0.001)

        self.assertAlmostEqual(var1["mean"], 10.0, delta=0.2)
        self.assertLess(var1["relative_half_width"], 0.1)

    def test_equilibration_failure_invalid_truncation(self):
        """Test CRError when MSER returns truncation point > half the data size."""

        # a linear trajectory
        def linear_trend_trajectory(step) -> np.ndarray:
            return np.linspace(0, 1000, step) + self.rng.normal(0, 1, step)

        with self.assertRaises(CRError) as cm:
            cr.run_length_control(
                get_trajectory=linear_trend_trajectory,
                initial_run_length=1000,
                maximum_run_length=5000,
                maximum_equilibration_step=2000,
                relative_accuracy=0.05,
            )
        emsg = str(cm.exception)
        self.assertIn("truncation point", emsg)
        self.assertIn("returned by MSER", emsg)

    def test_equilibration_failure_raises_error(self):
        """Test CRError raise when equilibration is not detected within limits."""
        self.setUp()  # reset state once more, deliberate, not redundant
        trajectory_state = {"current": 0}

        # a two-phase trajectory with args
        def two_phase_with_args_trajectory(step: int, args: dict) -> np.ndarray:
            idx = np.arange(args["current"], args["current"] + step)
            args["current"] += step

            plateau_end = 50
            ramp_slope = 0.0002
            base_value = 10.0

            ramp_delta = np.maximum(0, idx - plateau_end + 1)
            data = base_value + (ramp_slope * ramp_delta)
            data += 0.1 * self.rng.randn(step)

            return data

        with self.assertRaises(CRError) as cm:
            cr.run_length_control(
                get_trajectory=two_phase_with_args_trajectory,
                get_trajectory_args=trajectory_state,
                initial_run_length=500,
                run_length_factor=1.2,
                maximum_run_length=2000,
                maximum_equilibration_step=200,
                relative_accuracy=0.05,
                minimum_correlation_time=1,
            )
        emsg = str(cm.exception)
        self.assertIn("truncation point = ", emsg)
        self.assertIn(
            "returned by MSER + Integrated Autocorrelation Time refinement", emsg
        )
        self.assertIn(">= maximum_equilibration_step (200)", emsg)
        self.assertIn("is therefore invalid", emsg)

    def test_equilibration_failure_per_variable_messages(self):
        """Test per-variable error messages in _check_equilibration_step when only some variables fail."""

        # variable 0 never equilibrates, variable 1 equilibrates immediately
        def mixed_trajectory(step: int) -> np.ndarray:
            drift = np.arange(step) * 0.5
            steady = np.full(step, 10.0)
            return np.vstack([drift, steady])

        with self.assertRaises(cr.CRError) as ctx:
            cr.run_length_control(
                get_trajectory=mixed_trajectory,
                number_of_variables=2,
                initial_run_length=200,
                maximum_run_length=500,
                maximum_equilibration_step=100,  # var-0 will violate
                relative_accuracy=0.1,
                dump_trajectory=False,
            )
        emsg = str(ctx.exception)
        # message must contain the per-variable prefix
        self.assertIn("for variable number 1", emsg)
        self.assertIn("for variable number 2", emsg)
        self.assertIn(">= maximum_equilibration_step", emsg)

    def test_population_chi_square_test_only(self):
        """Test chi-square population test when only standard deviation is provided."""

        def normal_trajectory(step: int) -> np.ndarray:
            return self.rng.normal(5.0, 2.0, step)  # true std = 2.0

        converged = cr.run_length_control(
            get_trajectory=normal_trajectory,
            number_of_variables=1,
            initial_run_length=1000,
            maximum_run_length=5000,
            relative_accuracy=0.1,
            population_mean=None,  # skips t-test
            population_standard_deviation=2.0,  # triggers chi-square test
            confidence_coefficient=0.95,
            fp=None,
        )

        self.assertTrue(converged)

    def test_population_t_test_only(self):
        """Test t-test population hypothesis when only mean is provided."""

        def normal_trajectory(step: int) -> np.ndarray:
            return self.rng.normal(7.0, 1.5, step)  # true mean = 7.0

        converged = cr.run_length_control(
            get_trajectory=normal_trajectory,
            number_of_variables=1,
            initial_run_length=1000,
            maximum_run_length=5000,
            relative_accuracy=0.1,
            population_mean=7.0,  # triggers t-test
            population_standard_deviation=None,  # skips chi-square test
            confidence_coefficient=0.95,
            fp=None,
        )

        self.assertTrue(converged)

    def test_make_variable_list_string_scalar(self):
        """Test _make_variable_list repeats non-numeric scalar strings."""

        from kim_convergence.run_length_control._variable_list_factory import (
            _make_variable_list,
        )

        out = _make_variable_list("gamma", 3)
        self.assertEqual(out, ["gamma", "gamma", "gamma"])

    def test_make_variable_list_tuple_input(self):
        """Test _make_variable_list accepts tuple input of correct length."""

        from kim_convergence.run_length_control._variable_list_factory import (
            _make_variable_list,
        )

        out = _make_variable_list((0.1, None, 0.2), 3)
        self.assertEqual(out, [0.1, None, 0.2])

    def test_heidel_welch_ucl_method(self):
        """Test convergence using the Heidelberger-Welch UCL approximation method."""

        def constant_trajectory(step: int) -> np.ndarray:
            # Tiny noise to give Heidelberger-Welch something to work with
            return np.full(step, 4.0) + self.rng.normal(0.0, 1e-8, step)

        converged = cr.run_length_control(
            get_trajectory=constant_trajectory,
            number_of_variables=1,
            initial_run_length=100,
            maximum_run_length=500,
            relative_accuracy=0.1,
            confidence_interval_approximation_method="heidel_welch",
            heidel_welch_number_points=30,
            confidence_coefficient=0.95,
            fp=None,
        )

        self.assertTrue(converged)

    def test_absolute_accuracy_below_minimum(self):
        """Test CRError when absolute_accuracy is below the minimum threshold (relative_accuracy=None)."""

        def constant_trajectory(step: int) -> np.ndarray:
            return np.full(step, 1.0)

        with self.assertRaises(cr.CRError) as ctx:
            cr.run_length_control(
                get_trajectory=constant_trajectory,
                number_of_variables=1,
                initial_run_length=50,
                maximum_run_length=200,
                relative_accuracy=None,
                absolute_accuracy=1e-12,  # Below _DEFAULT_MIN_ABSOLUTE_ACCURACY
                fp=None,
            )

        emsg = str(ctx.exception)
        self.assertIn("absolute_accuracy", emsg)
        self.assertIn("must be greater than or equal", emsg)
        # Optional: check that the minimum value is mentioned
        self.assertIn("0.0001", emsg)  # assuming _DEFAULT_MIN_ABSOLUTE_ACCURACY = 1e-4

    def test_truncated_series_returns_view(self):
        """Test that _truncated_series returns a memory-shared view, not a copy."""

        from kim_convergence.run_length_control._equilibration import _truncated_series

        tsd = np.arange(20).reshape(2, 10)  # 2 variables, 10 steps each
        view = _truncated_series(tsd, ndim=2, truncate_index=3, var_idx=0)

        view[0] = -999
        self.assertEqual(tsd[0, 3], -999)  # Modification affects original array

    def test_equilibrated_series_returns_view(self):
        """Test that _equilibrated_series returns a memory-shared view, not a copy."""

        from kim_convergence.run_length_control._convergence import _equilibrated_series

        tsd = np.arange(10, dtype=float)
        view = _equilibrated_series(tsd, ndim=1, equilibration_step=4, var_idx=0)

        view[0] = -888.0
        self.assertEqual(tsd[4], -888.0)  # Modification affects original array

    def test_maximum_equilibration_step_upper_bound(self):
        """Test CRError when maximum_equilibration_step is not strictly less than maximum_run_length."""

        def constant_trajectory(step: int) -> np.ndarray:
            return np.ones(step)

        with self.assertRaises(cr.CRError) as ctx:
            cr.run_length_control(
                get_trajectory=constant_trajectory,
                number_of_variables=1,
                maximum_run_length=100,
                maximum_equilibration_step=100,  # Must be < maximum_run_length
                relative_accuracy=0.1,
                fp=None,
            )

        emsg = str(ctx.exception)
        self.assertIn("maximum_equilibration_step", emsg)
        self.assertIn("smaller than or equal", emsg)
        self.assertIn("99", emsg)

    def test_batch_size_lower_bound(self):
        """Test CRError when batch_size is zero (invalid for MSER)."""

        def constant_trajectory(step: int) -> np.ndarray:
            return np.ones(step)

        with self.assertRaises(cr.CRError) as ctx:
            cr.run_length_control(
                get_trajectory=constant_trajectory,
                batch_size=0,  # Triggers mser_m validation
                relative_accuracy=0.1,
            )

        emsg = str(ctx.exception)
        self.assertIn("batch_size", emsg)
        self.assertIn("< 1", emsg)  # Phrase used in mser_m error
        self.assertIn("not valid", emsg)

    def test_trajectory_wrong_shape_first_call(self):
        """Test CRError when get_trajectory returns wrong shape on initial acquisition."""

        def bad_trajectory(step: int) -> np.ndarray:
            # Returns 3 variables instead of the requested 2
            return np.ones((3, step))

        with self.assertRaises(cr.CRError) as ctx:
            cr.run_length_control(
                get_trajectory=bad_trajectory,
                number_of_variables=2,
                initial_run_length=50,
                relative_accuracy=0.1,  # Required argument
                fp=None,
            )

        emsg = str(ctx.exception)
        self.assertIn("wrong number of variables", emsg)
        self.assertIn("3", emsg)
        self.assertIn("2", emsg)

    def test_get_trajectory_raises_exception(self):
        """Test CRError when get_trajectory raises an exception during acquisition."""

        def buggy_trajectory(step: int) -> np.ndarray:
            raise ValueError("simulator crashed")

        with self.assertRaises(cr.CRError) as ctx:
            cr.run_length_control(
                get_trajectory=buggy_trajectory,
                initial_run_length=10,
                relative_accuracy=0.1,  # Required argument
            )

        emsg = str(ctx.exception)
        self.assertIn("failed to get the time-series data", emsg)
        self.assertIn("simulation", emsg)  # From the message text
        self.assertIn("10", emsg)  # The requested step count

    def test_non_finite_values_after_extension(self):
        """Test CRError when get_trajectory returns non-finite values in an extended segment."""

        trajectory_state = {"call_count": 0}

        def delayed_nan_trajectory(step: int, args: dict) -> np.ndarray:
            args["call_count"] += 1
            if args["call_count"] == 1:
                return 10 * self.rng.randn(step)  # First call: good data
            else:
                return np.full(step, np.nan)  # Subsequent calls: bad data

        with self.assertRaises(cr.CRError) as ctx:
            cr.run_length_control(
                get_trajectory=delayed_nan_trajectory,
                get_trajectory_args=trajectory_state,
                initial_run_length=20,
                run_length_factor=2.0,
                maximum_run_length=200,
                relative_accuracy=0.1,
                fp=None,
            )

        emsg = str(ctx.exception)
        self.assertIn("non-finite", emsg)
        self.assertIn("not number", emsg)  # From _get_trajectory message

    def test_dump_trajectory_and_report_to_files(self):
        """Test simultaneous dumping of trajectory and JSON report to separate files."""

        def constant_trajectory(step: int) -> np.ndarray:
            return np.full(step, 77.0)

        with tempfile.NamedTemporaryFile(suffix=".edn", delete=False) as f:
            traj_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report_file = f.name

        try:
            # Open report file and keep it open during the call
            with open(report_file, "w") as report_fp:
                converged = cr.run_length_control(
                    get_trajectory=constant_trajectory,
                    initial_run_length=100,
                    maximum_run_length=500,
                    relative_accuracy=0.1,
                    dump_trajectory=True,
                    dump_trajectory_fp=traj_file,
                    fp=report_fp,  # Pass the open file object
                    fp_format="json",
                )

            self.assertTrue(converged)

            # Now safe to read — file was closed by 'with'
            with open(report_file) as f:
                report = json.load(f)
            self.assertAlmostEqual(report["mean"], 77.0)

            # Trajectory file
            with open(traj_file) as f:
                dumped_traj = kim_edn.load(f)
            self.assertTrue(np.allclose(np.array(dumped_traj), 77.0))

        finally:
            for filepath in (traj_file, report_file):
                if os.path.exists(filepath):
                    os.unlink(filepath)

    def test_initial_run_too_short_for_mser(self):
        """Test convergence when initial_run_length is too short for MSER (forces extension)."""

        def constant_trajectory(step: int) -> np.ndarray:
            return np.ones(step)

        converged = cr.run_length_control(
            get_trajectory=constant_trajectory,
            initial_run_length=5,  # Too short: MSER requires more points
            batch_size=5,  # Makes minimum data requirement higher
            maximum_run_length=500,
            relative_accuracy=0.1,
            fp=None,
        )

        self.assertTrue(converged)  # Must extend trajectory and still converge

    def test_single_variable_2d_shape(self):
        """Test single variable accepts 2D trajectory input with shape (1, n)."""

        def single_variable_2d(step: int) -> np.ndarray:
            return np.full((1, step), 3.3)

        with self.assertRaises(cr.CRError) as ctx:
            cr.run_length_control(
                get_trajectory=single_variable_2d,
                number_of_variables=1,
                initial_run_length=100,
                maximum_run_length=500,
                relative_accuracy=0.1,
                fp=None,
            )

        emsg = str(ctx.exception)
        self.assertIn("get_trajectory", emsg)
        self.assertIn("function has a wrong dimension of 2 != 1", emsg)

    def test_get_run_length_various_scenarios(self):
        """Comprehensive test of _get_run_length behavior across key cases."""
        from kim_convergence.run_length_control._run_length import _get_run_length

        test_cases = [
            # (run_length, factor, total, max_total, expected, description)
            (100, 1.5, 500, 2000, 150, "standard growth with room"),
            (200, 1.8, 1700, 2000, 300, "growth capped by remaining budget"),
            (100, 2.0, 2000, 2000, 0, "already at maximum -> stop"),
            (100, 2.0, 2100, 2000, 0, "beyond maximum -> stop"),
            (500, 0.0, 300, 10000, 0, "factor = 0 -> immediate stop"),
            (200, 0.6, 400, 2000, 120, "shrinking (factor < 1)"),
            (10, 0.05, 800, 1000, 1, "very small factor -> clamped to min 1"),
            (120, 1.0, 800, 2000, 120, "factor = 1 -> no change"),
            (300, 1.0, 1850, 2000, 150, "factor = 1 but capped by remaining"),
            (100, 2, 300, 2000, 200, "integer factor (same as float)"),
            (100, 2.0, 300, 2000, 200, "float factor — must match integer case"),
        ]

        for run_length, factor, total, max_total, expected, desc in test_cases:
            with self.subTest(
                description=desc,
                run_length=run_length,
                factor=factor,
                total=total,
                max_total=max_total,
            ):
                result = _get_run_length(
                    run_length=run_length,
                    run_length_factor=factor,
                    total_run_length=total,
                    maximum_run_length=max_total,
                )
                self.assertEqual(
                    result,
                    expected,
                    f"Expected {expected} but got {result} for: {desc}",
                )
