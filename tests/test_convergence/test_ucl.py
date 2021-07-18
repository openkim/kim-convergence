"""Test ucl module."""
from os import truncate
import unittest
import numpy as np

try:
    import convergence as cr
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


def first_order_autoregressive_process(*,
                                       mu=100,
                                       rho=0.995,
                                       x0=0,
                                       n=25000,
                                       rng=np.random.RandomState(12345)):
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
        except CVGError:
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
            (np.arange(n // 10)[::-1] / float(n // 10),
             np.zeros(n - n // 10)))

        x += y

        test_passed = True

        try:
            truncated, truncated_index = \
                ucl_base.estimate_equilibration_length(x)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertTrue(truncated)
        self.assertEqual(ucl_base.si, 1.0)

        x_cut = x[truncated_index:]

        try:
            ucl_base.set_indices(x_cut)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertEqual(x_cut.size, ucl_base.indices.size)

        ucl = ucl_base.ucl(x_cut)
        self.assertEqual(ucl, 1e100)

        self.assertRaises(TypeError, ucl_base.ci, x_cut)

    def test_set_heidel_welch_constants(self):
        """Test set_heidel_welch_constants function."""
        test_passed = True

        try:
            # Initialize the HeidelbergerWelch object
            heidel_welch = cr.HeidelbergerWelch()
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        heidel_welch_set, \
            heidel_welch_k, \
            heidel_welch_n, \
            heidel_welch_p, \
            a_matrix, \
            a_matrix_1_inv, \
            a_matrix_2_inv, \
            a_matrix_3_inv, \
            heidel_welch_c1_1, \
            heidel_welch_c1_2, \
            heidel_welch_c1_3, \
            heidel_welch_c2_1, \
            heidel_welch_c2_2, \
            heidel_welch_c2_3, \
            tm_1, \
            tm_2, \
            tm_3 = heidel_welch.get_heidel_welch_constants()

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
                confidence_coefficient=0.95, heidel_welch_number_points=50)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        heidel_welch_set, \
            heidel_welch_k, \
            heidel_welch_n, \
            heidel_welch_p, \
            a_matrix, \
            a_matrix_1_inv, \
            a_matrix_2_inv, \
            a_matrix_3_inv, \
            heidel_welch_c1_1, \
            heidel_welch_c1_2, \
            heidel_welch_c1_3, \
            heidel_welch_c2_1, \
            heidel_welch_c2_2, \
            heidel_welch_c2_3, \
            tm_1, \
            tm_2, \
            tm_3 = heidel_welch.get_heidel_welch_constants()

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
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.0,
            heidel_welch_number_points=50)

        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=1.0,
            heidel_welch_number_points=50)

        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points=0)

        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points=10)

        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points=50.0)

        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points=-10.0)

        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.95,
            heidel_welch_number_points='50')

        test_passed = True

        try:
            # Initialize the HeidelbergerWelch object
            heidel_welch.set_heidel_welch_constants(
                confidence_coefficient=0.95, heidel_welch_number_points=50)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertTrue(heidel_welch.is_heidel_welch_set())

        heidel_welch_k, heidel_welch_n, \
            heidel_welch_p = heidel_welch.get_heidel_welch_knp()

        self.assertTrue(heidel_welch_k == 50)
        self.assertTrue(heidel_welch_n == 200)
        self.assertTrue(heidel_welch_p == 0.95)

        heidel_welch_c1_1, heidel_welch_c1_2, \
            heidel_welch_c1_3 = heidel_welch.get_heidel_welch_c1()

        self.assertAlmostEqual(heidel_welch_c1_1, 0.974, places=3)
        self.assertAlmostEqual(heidel_welch_c1_2, 0.941, places=3)
        # In the paper it is rounded up to 0.895
        self.assertAlmostEqual(heidel_welch_c1_3, 0.894, places=3)

        heidel_welch_c2_1, heidel_welch_c2_2, \
            heidel_welch_c2_3 = heidel_welch.get_heidel_welch_c2()

        self.assertTrue(heidel_welch_c2_1 == 37)
        self.assertTrue(heidel_welch_c2_2 == 16)
        self.assertTrue(heidel_welch_c2_3 == 8)

        heidel_welch.set_heidel_welch_constants(
            confidence_coefficient=0.95, heidel_welch_number_points=25)

        heidel_welch_k, heidel_welch_n, \
            heidel_welch_p = heidel_welch.get_heidel_welch_knp()

        self.assertTrue(heidel_welch_k == 25)
        self.assertTrue(heidel_welch_n == 100)
        self.assertTrue(heidel_welch_p == 0.95)

        heidel_welch_c1_1, heidel_welch_c1_2, \
            heidel_welch_c1_3 = heidel_welch.get_heidel_welch_c1()

        self.assertAlmostEqual(heidel_welch_c1_1, 0.948, places=3)
        # In the paper it is rounded up to 0.882
        self.assertAlmostEqual(heidel_welch_c1_2, 0.881, places=3)
        # In the paper it is rounded up to 0.784
        self.assertAlmostEqual(heidel_welch_c1_3, 0.783, places=3)

        heidel_welch_c2_1, heidel_welch_c2_2, \
            heidel_welch_c2_3 = heidel_welch.get_heidel_welch_c2()

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
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            # Initialize the HeidelbergerWelch object
            heidel_welch.set_heidel_welch_constants(
                confidence_coefficient=0.95, heidel_welch_number_points=50)
        except CVGError:
            test_passed = False

        x = np.arange(100.)

        self.assertRaises(CVGError, heidel_welch.ucl, x.reshape(5, 20))

        # input data points are not sufficient
        x = np.arange(10.)

        self.assertRaises(CVGError, heidel_welch.ucl, x)

        rng = np.random.RandomState(12345)

        x = np.ones(1000) * 10 + (rng.random_sample(1000) - 0.5)

        test_passed = True

        try:
            truncated, truncated_index = \
                heidel_welch.estimate_equilibration_length(x)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertTrue(truncated)
        self.assertEqual(truncated_index, 0)
        self.assertEqual(heidel_welch.si, 1.0)

        test_passed = True

        try:
            upper_confidence_limit_1 = heidel_welch.ucl(x)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertAlmostEqual(heidel_welch.mean, 10, places=2)

        try:
            lower, upper = heidel_welch.ci(x)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertEqual(heidel_welch.mean - upper_confidence_limit_1, lower)
        self.assertEqual(heidel_welch.mean + upper_confidence_limit_1, upper)

        try:
            relative_half_width_estimate = \
                heidel_welch.relative_half_width_estimate(x)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertEqual(relative_half_width_estimate,
                         upper_confidence_limit_1 / heidel_welch.mean)

        try:
            upper_confidence_limit_2 = cr.heidelberger_welch_ucl(x)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertEqual(upper_confidence_limit_1, upper_confidence_limit_2)

        try:
            upper_confidence_limit_2 = cr.heidelberger_welch_ucl(
                x, obj=heidel_welch)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertEqual(upper_confidence_limit_1, upper_confidence_limit_2)

        try:
            cr.heidelberger_welch_ci(x)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            cr.heidelberger_welch_ci(x, obj=heidel_welch)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            cr.heidelberger_welch_relative_half_width_estimate(x)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            cr.heidelberger_welch_relative_half_width_estimate(
                x, obj=heidel_welch)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

    def test_UncorrelatedSamples(self):
        """Test UncorrelatedSamples class."""
        test_passed = True

        try:
            # Initialize the UncorrelatedSamples object
            usamples = cr.UncorrelatedSamples()
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertIsNone(usamples.mean)
        self.assertIsNone(usamples.std)
        self.assertIsNone(usamples.si)
        self.assertIsNone(usamples.indices)

        x = np.arange(100.)

        self.assertRaises(CVGError, usamples.ucl, x.reshape(5, 20))

        self.assertRaises(CVGError, usamples.ucl, x, confidence_coefficient=0)
        self.assertRaises(CVGError, usamples.ucl, x, confidence_coefficient=1)

        # input data points are not sufficient
        x = np.arange(4.)

        self.assertRaises(CVGError, usamples.ucl, x)

        rng = np.random.RandomState(12345)

        x = np.ones(1000) * 10 + (rng.random_sample(1000) - 0.5)

        test_passed = True

        try:
            usamples.ucl(x)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertAlmostEqual(usamples.mean, x.mean())
        self.assertAlmostEqual(usamples.std, x.std())
        self.assertTrue(np.size(usamples.indices) == 1000)

        y = np.array((x[0], x[0], x[0], x[0], x[0], x[0], x[0], x[0]))

        try:
            usamples.ucl(y)
        except CVGError:
            test_passed = False

        self.assertFalse(test_passed)

        test_passed = True

        indices = np.arange(y.size)

        try:
            ucl = usamples.ucl(y, uncorrelated_sample_indices=indices)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertAlmostEqual(ucl, 0)

        self.assertAlmostEqual(usamples.mean, x[0])
        self.assertAlmostEqual(usamples.std, 0)
        self.assertTrue(np.all(indices == usamples.indices))

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
        self.assertRaises(CVGError, mser.ucl, x.reshape(5, 20))

        # nan in the input
        if n > 10:
            x[10] = np.nan

        self.assertRaises(CVGError, mser.ucl, x)

        rng = np.random.RandomState(12345)

        n = 1000
        x = np.arange(10.)
        y = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        x = np.concatenate((x, y))

        truncated, truncated_i = mser.estimate_equilibration_length(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i >= 10)
        self.assertAlmostEqual(mser.si, 1.0)

        test_passed = True

        try:
            mser.ucl(x[truncated_i:])
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertAlmostEqual(mser.mean, 10, places=2)
        self.assertTrue(mser.std < np.std(x[truncated_i:]))

        rng = np.random.RandomState(12345)

        n = 4000
        x = np.arange(0, 10, 10. / (n / 4))
        x += (rng.random_sample(x.size) - 0.5)
        y = np.ones(n) * 10 + rng.randn(n)
        x = np.concatenate((x, y))

        truncated, truncated_i = mser.estimate_equilibration_length(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i < n // 2)

        test_passed = True

        try:
            ucl = mser.ucl(x[truncated_i:])
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertAlmostEqual(mser.mean, 10, places=1)

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
        self.assertRaises(CVGError, mser.ucl, x.reshape(5, 20))

        # nan in the input
        if n > 10:
            x[10] = np.nan

        self.assertRaises(CVGError, mser.ucl, x)

        rng = np.random.RandomState(12345)

        n = 1000
        x = np.arange(10.)
        y = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        x = np.concatenate((x, y))

        truncated, truncated_i = mser.estimate_equilibration_length(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i >= 10)
        self.assertAlmostEqual(mser.si, 1.0)

        test_passed = True

        try:
            mser.ucl(x[truncated_i:])
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertAlmostEqual(mser.mean, 10, places=2)
        self.assertAlmostEqual(mser.std, np.std(x[truncated_i:]))

        rng = np.random.RandomState(12345)

        n = 4000
        x = np.arange(0, 10, 10. / (n / 4))
        x += (rng.random_sample(x.size) - 0.5)
        y = np.ones(n) * 10 + rng.randn(n)
        x = np.concatenate((x, y))

        truncated, truncated_i = mser.estimate_equilibration_length(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i < n // 2)

        test_passed = True

        try:
            ucl = mser.ucl(x[truncated_i:])
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertAlmostEqual(mser.mean, 10, places=1)

    def test_N_SKART(self):
        """Test N_SKART class."""
        skart = cr.N_SKART()

        self.assertIsNone(skart.indices)
        self.assertIsNone(skart.si)
        self.assertIsNone(skart.mean)
        self.assertIsNone(skart.std)

        x = np.arange(100)

        self.assertRaises(CVGError,
                          skart.estimate_equilibration_length,
                          x.reshape(5, 20))

        # input data points are not sufficient
        self.assertRaises(CVGError, skart.estimate_equilibration_length, x)
        self.assertRaises(CVGError, skart.ucl, x)

        rng = np.random.RandomState(12345)

        n = 100000
        x = np.arange(0, 10, 10. / (n / 100))
        x += (rng.random_sample(x.size) - 0.5)
        y = np.ones(n) * 10 + rng.randn(n)
        x = np.concatenate((x, y))

        truncated, truncated_i = skart.estimate_equilibration_length(x)

        self.assertTrue(truncated)
        self.assertTrue(truncated_i < n // 2)

        test_passed = True

        try:
            ucl = skart.ucl(x[truncated_i:])
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)
        self.assertAlmostEqual(skart.mean, 10, places=1)

    def test_comparison(self):
        """Test comparison."""
        x = first_order_autoregressive_process()

        print()

        truncated, truncated_index = cr.mser_m(x)
        self.assertTrue(truncated)

        print()
        print('truncated={}, truncated_index={}'.format(
            truncated, truncated_index))

        y = x[truncated_index:]

        equilibration_index, si_value = cr.estimate_equilibration_length(y)

        print()
        print('equilibration_index={}, si_value={}'.format(
            equilibration_index, si_value))

        z = y[equilibration_index:]

        heidel_welch = cr.HeidelbergerWelch()
        heidel_welch.set_heidel_welch_constants()
        ucl_heidel_welch = heidel_welch.ucl(z)

        print()
        print('ucl_heidel_welch={}'.format(ucl_heidel_welch))
        print('mean={}'.format(heidel_welch.mean))
        print('std={}'.format(heidel_welch.std))

        usamples = cr.UncorrelatedSamples()
        ucl_usamples = usamples.ucl(z)

        print()
        print('ucl_usamples={}'.format(ucl_usamples))
        print('mean={}'.format(usamples.mean))
        print('std={}'.format(usamples.std))

        mser = cr.MSER_m()
        ucl_mser = mser.ucl(z)

        print()
        print('ucl_mser={}'.format(ucl_mser))
        print('mean={}'.format(mser.mean))
        print('std={}'.format(mser.std))

        mser_y = cr.MSER_m_y()
        ucl_mser_y = mser_y.ucl(z)

        print()
        print('ucl_mser_y={}'.format(ucl_mser_y))
        print('mean={}'.format(mser_y.mean))
        print('std={}'.format(mser_y.std))

        skart = cr.N_SKART()
        ucl_skart = skart.ucl(z)

        print()
        print('ucl_skart={}'.format(ucl_skart))
        print('mean={}'.format(skart.mean))
        print('std={}'.format(skart.std))
