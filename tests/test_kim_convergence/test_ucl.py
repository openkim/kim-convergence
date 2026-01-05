"""Test ucl module."""

import unittest
import numpy as np
from typing import cast

try:
    import kim_convergence as cr
except Exception as e:  # intentional catch-all
    raise RuntimeError("Failed to import `kim-convergence` utility module") from e

from kim_convergence import CRError


def first_order_autoregressive_process(
    *, mu=100, rho=0.995, x0=0, n=25000, rng=np.random.RandomState(12345)
):
    r"""Return the first-order autoregressive process array of n samples.

    The first-order autoregressive process defined via the relation,

    .. math::

        X_k = \mu + \rho (X_{k-1} - \mu) + \epsilon_k,~~\text{for~}k = 1,2,\cdots,

    where :math:`\mu=100`, :math:`\rho=0.995`, and the error
    :math:`\{\epsilon_k\}` are i.i.d. standard normal random variables, and the
    initial state is :math:`X_0=0`. The steady-state distribution of this
    process is normal with mean :math:`\mu` and standard deviation
    :math:`\sigma = \frac{1}{\sqrt{1-\rho^2}} = 10.0125`.
    The initial state is located about 10 standard deviations below the steady
    state mean.

    For 3.75% Precision and Nominal 95% CIs

    SPSTS
        Avg. sample size = 85,857
        Avg. CI half-length = 1.507
        St.Dev. CI half-length = 0.472

    Skart
        Avg. sample size = 21,947
        Avg. CI half-length = 3.172
        St.Dev. CI half-length = 0.363

    ASAP3
        Avg. sample size = 41,208
        Avg. CI half-length = 2.820
        St.Dev. CI half-length = 0.507

    Args:
        mu (float, optional): mean mu. (default: 100)
        rho (float, optional): rho. (default: 0.995)
        x0 (float, optional): the initial state. (default: 0.0)
        n (int, optional): number of samples. (default: 25000)
        rng (`np.random.RandomState()`, optional): random number generator.
            (default: np.random.RandomState(12345))

    Returns:
        1darray: x (first-order autoregressive process array)

    """
    x = np.ones(n) * mu
    x[0] = x0
    for k in range(1, n):
        x[k] += rho * (x[k - 1] - mu) + rng.randn()
    return x


class TestUCLModule(unittest.TestCase):
    """Test ucl module components."""

    def test_ucl_base(self):
        """Test Upper Confidence Limit base class."""
        test_passed = True

        try:
            # Initialize the HeidelbergerWelch object
            ucl_base = cr.ucl.UCLBase()
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertIsNone(ucl_base.mean)
        self.assertIsNone(ucl_base.std)
        self.assertIsNone(ucl_base.si)
        self.assertIsNone(ucl_base.indices)

        rng = np.random.RandomState(12345)
        n = 1000
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        y = np.concatenate(
            (np.arange(n // 10)[::-1] / float(n // 10), np.zeros(n - n // 10))
        )

        x += y

        test_passed = True

        try:
            truncated, truncated_index = ucl_base.estimate_equilibration_length(x)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertTrue(truncated)
        self.assertEqual(ucl_base.si, 1.0)

        x_cut = x[truncated_index:]

        try:
            ucl_base.set_indices(x_cut)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertIsInstance(ucl_base.indices, np.ndarray)
        assert isinstance(ucl_base.indices, np.ndarray)  # keeps mypy happy
        self.assertEqual(x_cut.size, ucl_base.indices.size)

        self.assertRaises(NotImplementedError, ucl_base.ucl, x_cut)
        self.assertRaises(NotImplementedError, ucl_base.ci, x_cut)
        self.assertRaises(
            NotImplementedError, ucl_base.relative_half_width_estimate, x_cut
        )

        # zero-variance (constant) case
        x = np.ones(n) * 10
        ucl = ucl_base.ucl(x)
        self.assertAlmostEqual(ucl, 0)
        self.assertAlmostEqual(cast(float, ucl_base.mean), 10)
        self.assertAlmostEqual(cast(float, ucl_base.std), 0)
        self.assertEqual(ucl_base.sample_size, n)
        self.assertAlmostEqual(cast(float, ucl_base.si), n)
        self.assertEqual(ucl_base.indices.size, 1)

    def test_set_heidel_welch_constants(self):
        """Test set_heidel_welch_constants function."""
        test_passed = True

        try:
            # Initialize the HeidelbergerWelch object
            heidel_welch = cr.HeidelbergerWelch()
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        (
            heidel_welch_set,
            heidel_welch_k,
            heidel_welch_n,
            heidel_welch_p,
            a_matrix,
            a_matrix_1_inv,
            a_matrix_2_inv,
            a_matrix_3_inv,
            heidel_welch_c1_1,
            heidel_welch_c1_2,
            heidel_welch_c1_3,
            heidel_welch_c2_1,
            heidel_welch_c2_2,
            heidel_welch_c2_3,
            tm_1,
            tm_2,
            tm_3,
        ) = heidel_welch.get_heidel_welch_constants()

        self.assertFalse(heidel_welch_set)
        self.assertIsNone(heidel_welch_k)
        self.assertIsNone(heidel_welch_n)
        self.assertIsNone(heidel_welch_p)
        self.assertIsNone(a_matrix)
        self.assertIsNone(a_matrix_1_inv)
        self.assertIsNone(a_matrix_2_inv)
        self.assertIsNone(a_matrix_3_inv)
        self.assertIsNone(heidel_welch_c1_1)
        self.assertIsNone(heidel_welch_c1_2)
        self.assertIsNone(heidel_welch_c1_3)
        self.assertIsNone(heidel_welch_c2_1)
        self.assertIsNone(heidel_welch_c2_2)
        self.assertIsNone(heidel_welch_c2_3)
        self.assertIsNone(tm_1)
        self.assertIsNone(tm_2)
        self.assertIsNone(tm_3)

        self.assertIsNone(heidel_welch.mean)
        self.assertIsNone(heidel_welch.std)
        self.assertIsNone(heidel_welch.si)
        self.assertIsNone(heidel_welch.indices)

        test_passed = True

        try:
            # Initialize the HeidelbergerWelch object
            heidel_welch.set_heidel_welch_constants(
                confidence_coefficient=0.95, heidel_welch_number_points=50
            )
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        (
            heidel_welch_set,
            heidel_welch_k,
            heidel_welch_n,
            heidel_welch_p,
            a_matrix,
            a_matrix_1_inv,
            a_matrix_2_inv,
            a_matrix_3_inv,
            heidel_welch_c1_1,
            heidel_welch_c1_2,
            heidel_welch_c1_3,
            heidel_welch_c2_1,
            heidel_welch_c2_2,
            heidel_welch_c2_3,
            tm_1,
            tm_2,
            tm_3,
        ) = heidel_welch.get_heidel_welch_constants()

        self.assertTrue(heidel_welch_set)
        self.assertIsNotNone(heidel_welch_k)
        self.assertIsNotNone(heidel_welch_n)
        self.assertIsNotNone(heidel_welch_p)
        self.assertIsNotNone(a_matrix)
        self.assertIsNotNone(a_matrix_1_inv)
        self.assertIsNotNone(a_matrix_2_inv)
        self.assertIsNotNone(a_matrix_3_inv)
        self.assertIsNotNone(heidel_welch_c1_1)
        self.assertIsNotNone(heidel_welch_c1_2)
        self.assertIsNotNone(heidel_welch_c1_3)
        self.assertIsNotNone(heidel_welch_c2_1)
        self.assertIsNotNone(heidel_welch_c2_2)
        self.assertIsNotNone(heidel_welch_c2_3)
        self.assertIsNotNone(tm_1)
        self.assertIsNotNone(tm_2)
        self.assertIsNotNone(tm_3)

        heidel_welch.unset_heidel_welch_constants()

        self.assertRaises(
            CRError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.0,
            heidel_welch_number_points=50,
        )

        self.assertRaises(
            CRError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=1.0,
            heidel_welch_number_points=50,
        )

        self.assertRaises(
            CRError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points=0,
        )

        self.assertRaises(
            CRError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points=10,
        )

        self.assertRaises(
            CRError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points=50.0,
        )

        self.assertRaises(
            CRError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points=-10.0,
        )

        self.assertRaises(
            CRError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points="50",
        )

        test_passed = True

        try:
            # Initialize the HeidelbergerWelch object
            heidel_welch.set_heidel_welch_constants(
                confidence_coefficient=0.95, heidel_welch_number_points=50
            )
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertTrue(heidel_welch.is_heidel_welch_set())

        heidel_welch_k, heidel_welch_n, heidel_welch_p = (
            heidel_welch.get_heidel_welch_knp()
        )

        self.assertTrue(heidel_welch_k == 50)
        self.assertTrue(heidel_welch_n == 200)
        self.assertTrue(heidel_welch_p == 0.95)

        heidel_welch_c1_1, heidel_welch_c1_2, heidel_welch_c1_3 = (
            heidel_welch.get_heidel_welch_c1()
        )

        self.assertAlmostEqual(heidel_welch_c1_1, 0.974, places=3)
        self.assertAlmostEqual(heidel_welch_c1_2, 0.941, places=3)
        # In the paper it is rounded up to 0.895
        self.assertAlmostEqual(heidel_welch_c1_3, 0.894, places=3)

        heidel_welch_c2_1, heidel_welch_c2_2, heidel_welch_c2_3 = (
            heidel_welch.get_heidel_welch_c2()
        )

        self.assertTrue(heidel_welch_c2_1 == 37)
        self.assertTrue(heidel_welch_c2_2 == 16)
        self.assertTrue(heidel_welch_c2_3 == 8)

        heidel_welch.set_heidel_welch_constants(
            confidence_coefficient=0.95, heidel_welch_number_points=25
        )

        heidel_welch_k, heidel_welch_n, heidel_welch_p = (
            heidel_welch.get_heidel_welch_knp()
        )

        self.assertTrue(heidel_welch_k == 25)
        self.assertTrue(heidel_welch_n == 100)
        self.assertTrue(heidel_welch_p == 0.95)

        heidel_welch_c1_1, heidel_welch_c1_2, heidel_welch_c1_3 = (
            heidel_welch.get_heidel_welch_c1()
        )

        self.assertAlmostEqual(heidel_welch_c1_1, 0.948, places=3)
        # In the paper it is rounded up to 0.882
        self.assertAlmostEqual(heidel_welch_c1_2, 0.881, places=3)
        # In the paper it is rounded up to 0.784
        self.assertAlmostEqual(heidel_welch_c1_3, 0.783, places=3)

        heidel_welch_c2_1, heidel_welch_c2_2, heidel_welch_c2_3 = (
            heidel_welch.get_heidel_welch_c2()
        )

        self.assertTrue(heidel_welch_c2_1 == 18)
        self.assertTrue(heidel_welch_c2_2 == 7)
        self.assertTrue(heidel_welch_c2_3 == 3)

        heidel_welch.unset_heidel_welch_constants()

    def test_HeidelbergerWelch(self):
        """Test HeidelbergerWelch class."""

        test_passed = True

        try:
            # Initialize the HeidelbergerWelch object
            heidel_welch = cr.HeidelbergerWelch()
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            # Initialize the HeidelbergerWelch object
            heidel_welch.set_heidel_welch_constants(
                confidence_coefficient=0.95, heidel_welch_number_points=50
            )
        except CRError:
            test_passed = False

        x = np.arange(100.0)

        self.assertRaises(CRError, heidel_welch.ucl, x.reshape(5, 20))

        # input data points are not sufficient
        x = np.arange(10.0)

        self.assertRaises(CRError, heidel_welch.ucl, x)

        rng = np.random.RandomState(12345)

        x = np.ones(1000) * 10 + (rng.random_sample(1000) - 0.5)

        test_passed = True

        try:
            truncated, truncated_index = heidel_welch.estimate_equilibration_length(x)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertTrue(truncated)
        self.assertEqual(truncated_index, 0)
        self.assertEqual(heidel_welch.si, 1.0)

        test_passed = True

        try:
            upper_confidence_limit_1 = heidel_welch.ucl(x)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertIsInstance(heidel_welch.mean, float)
        self.assertAlmostEqual(cast(float, heidel_welch.mean), 10, places=2)

        try:
            lower, upper = heidel_welch.ci(x)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertEqual(
            cast(float, heidel_welch.mean) - upper_confidence_limit_1, lower
        )
        self.assertEqual(
            cast(float, heidel_welch.mean) + upper_confidence_limit_1, upper
        )

        try:
            relative_half_width_estimate = heidel_welch.relative_half_width_estimate(x)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertEqual(
            relative_half_width_estimate,
            upper_confidence_limit_1 / cast(float, heidel_welch.mean),
        )

        try:
            upper_confidence_limit_2 = cr.heidelberger_welch_ucl(x)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertEqual(upper_confidence_limit_1, upper_confidence_limit_2)

        try:
            upper_confidence_limit_2 = cr.heidelberger_welch_ucl(x, obj=heidel_welch)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertEqual(upper_confidence_limit_1, upper_confidence_limit_2)

        try:
            cr.heidelberger_welch_ci(x)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            cr.heidelberger_welch_ci(x, obj=heidel_welch)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            cr.heidelberger_welch_relative_half_width_estimate(x)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            cr.heidelberger_welch_relative_half_width_estimate(x, obj=heidel_welch)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

    def test_UncorrelatedSamples(self):
        """Test UncorrelatedSamples class."""
        test_passed = True

        try:
            # Initialize the UncorrelatedSamples object
            usamples = cr.UncorrelatedSamples()
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertIsNone(usamples.mean)
        self.assertIsNone(usamples.std)
        self.assertIsNone(usamples.si)
        self.assertIsNone(usamples.indices)

        x = np.arange(100.0)

        self.assertRaises(CRError, usamples.ucl, x.reshape(5, 20))

        self.assertRaises(CRError, usamples.ucl, x, confidence_coefficient=0)
        self.assertRaises(CRError, usamples.ucl, x, confidence_coefficient=1)

        # input data points are not sufficient
        x = np.arange(4.0)

        self.assertRaises(CRError, usamples.ucl, x)

        rng = np.random.RandomState(12345)

        x = np.ones(1000) * 10 + (rng.random_sample(1000) - 0.5)

        test_passed = True

        try:
            usamples.ucl(x)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertAlmostEqual(usamples.mean, x.mean())
        self.assertAlmostEqual(usamples.std, x.std())
        self.assertTrue(np.size(cast(np.ndarray, usamples.indices)) == 1000)

        y = np.array(([x[0]] * 8))

        try:
            usamples.ucl(y)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        indices = np.arange(y.size)

        try:
            ucl = usamples.ucl(y, uncorrelated_sample_indices=indices)
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertAlmostEqual(ucl, 0)

        self.assertAlmostEqual(usamples.mean, x[0])
        self.assertAlmostEqual(cast(float, usamples.std), 0)
        self.assertTrue(np.all(np.array([0], dtype=int) == usamples.indices))

    def test_MSER_m(self):
        """Test MSER_m class."""
        mser = cr.MSER_m()

        self.assertIsNone(mser.indices)
        self.assertIsNone(mser.si)
        self.assertIsNone(mser.mean)
        self.assertIsNone(mser.std)

        n = 100
        x = np.ones(n)

        # x is not one dimensional
        self.assertRaises(CRError, mser.ucl, x.reshape(5, 20))

        # nan in the input
        if n > 10:
            x[10] = np.nan

        self.assertRaises(CRError, mser.ucl, x)

        rng = np.random.RandomState(12345)

        n = 1000
        x = np.arange(10.0)
        y = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        x = np.concatenate((x, y))

        truncated, truncated_i = mser.estimate_equilibration_length(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i >= 10)
        self.assertAlmostEqual(cast(float, mser.si), 1.0)

        test_passed = True

        try:
            mser.ucl(x[truncated_i:])
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertAlmostEqual(cast(float, mser.mean), 10, places=2)
        self.assertTrue(cast(float, mser.std) < np.std(x[truncated_i:]))

        rng = np.random.RandomState(12345)

        n = 4000
        x = np.arange(0, 10, 10.0 / (n / 4))
        x += rng.random_sample(x.size) - 0.5
        y = np.ones(n) * 10 + rng.randn(n)
        x = np.concatenate((x, y))

        truncated, truncated_i = mser.estimate_equilibration_length(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i < n // 2)

        test_passed = True

        try:
            mser.ucl(x[truncated_i:])
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertAlmostEqual(cast(float, mser.mean), 10, places=1)

    def test_MSER_m_y(self):
        """Test MSER_m_y class."""
        mser = cr.MSER_m_y()

        self.assertIsNone(mser.indices)
        self.assertIsNone(mser.si)
        self.assertIsNone(mser.mean)
        self.assertIsNone(mser.std)

        n = 100
        x = np.ones(n)

        # x is not one dimensional
        self.assertRaises(CRError, mser.ucl, x.reshape(5, 20))

        # nan in the input
        if n > 10:
            x[10] = np.nan

        self.assertRaises(CRError, mser.ucl, x)

        rng = np.random.RandomState(12345)

        n = 1000
        x = np.arange(10.0)
        y = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        x = np.concatenate((x, y))

        truncated, truncated_i = mser.estimate_equilibration_length(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i >= 10)
        self.assertAlmostEqual(cast(float, mser.si), 1.0)

        test_passed = True

        try:
            mser.ucl(x[truncated_i:])
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertAlmostEqual(cast(float, mser.mean), 10, places=2)
        self.assertAlmostEqual(cast(float, mser.std), np.std(x[truncated_i:]))

        rng = np.random.RandomState(12345)

        n = 4000
        x = np.arange(0, 10, 10.0 / (n / 4))
        x += rng.random_sample(x.size) - 0.5
        y = np.ones(n) * 10 + rng.randn(n)
        x = np.concatenate((x, y))

        truncated, truncated_i = mser.estimate_equilibration_length(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i < n // 2)

        test_passed = True

        try:
            mser.ucl(x[truncated_i:])
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertAlmostEqual(cast(float, mser.mean), 10, places=1)

    def test_N_SKART(self):
        """Test N_SKART class."""
        skart = cr.N_SKART()

        self.assertIsNone(skart.indices)
        self.assertIsNone(skart.si)
        self.assertIsNone(skart.mean)
        self.assertIsNone(skart.std)

        x = np.arange(100)

        self.assertRaises(
            CRError, skart.estimate_equilibration_length, x.reshape(5, 20)
        )

        # input data points are not sufficient
        self.assertRaises(CRError, skart.estimate_equilibration_length, x)
        self.assertRaises(CRError, skart.ucl, x)

        rng = np.random.RandomState(12345)

        n = 100000
        x = np.arange(0, 10, 10.0 / (n / 100))
        x += rng.random_sample(x.size) - 0.5
        y = np.ones(n) * 10 + rng.randn(n)
        x = np.concatenate((x, y))

        truncated, truncated_i = skart.estimate_equilibration_length(x)

        self.assertTrue(truncated)
        self.assertTrue(truncated_i < n // 2)

        test_passed = True

        try:
            skart.ucl(x[truncated_i:])
        except CRError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertAlmostEqual(cast(float, skart.mean), 10, places=1)

    def test_comparison(self):
        """Test comparison."""
        x = first_order_autoregressive_process()

        print("\nUCL Results on equilibrated autoregressive process:\n")

        truncated, truncated_index = cr.mser_m(x)
        self.assertTrue(truncated)
        print(f"{truncated=}, {truncated_index=}")

        y = x[truncated_index:]

        equilibration_index, si_value = cr.estimate_equilibration_length(y)
        print(f"{equilibration_index=}, {si_value=}\n")

        z = y[equilibration_index:]

        ucl_classes = [
            ("HeidelbergerWelch", cr.HeidelbergerWelch()),
            ("UncorrelatedSamples", cr.UncorrelatedSamples()),
            ("MSER_m", cr.MSER_m()),
            ("MSER_m_y", cr.MSER_m_y()),
            ("N_SKART", cr.N_SKART()),
        ]

        for name, ucl_obj in ucl_classes:
            if name == "HeidelbergerWelch":
                ucl_obj.set_heidel_welch_constants()

            ucl_value = ucl_obj.ucl(z)

            print(f"{name}:")
            print(f"  UCL   = {ucl_value}")
            print(f"  mean  = {ucl_obj.mean}")
            print(f"  std   = {ucl_obj.std}")
            print()
