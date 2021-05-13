"""Test ucl module."""
import unittest
import numpy as np
from numpy.testing._private.utils import assert_equal

try:
    import convergence as cr
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TestUCLModule(unittest.TestCase):
    """Test ucl module components."""

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

        self.assertRaises(CVGError, heidel_welch.ucl,
                          x.reshape(5, 20))

        # input data points are not sufficient
        x = np.arange(10.)

        self.assertRaises(CVGError, heidel_welch.ucl, x)

        rng = np.random.RandomState(12345)

        x = np.ones(1000) * 10 + (rng.random_sample(1000) - 0.5)

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

        self.assertEqual(heidel_welch.mean - upper_confidence_limit_1,
                         lower)
        self.assertEqual(heidel_welch.mean + upper_confidence_limit_1,
                         upper)

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
            # Initialize the HeidelbergerWelch object
            usamples = cr.UncorrelatedSamples()
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertIsNone(usamples.mean)
        self.assertIsNone(usamples.std)
        self.assertIsNone(usamples.si)
        self.assertIsNone(usamples.indices)