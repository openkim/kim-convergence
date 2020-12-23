"""Test ucl module."""
import unittest
import numpy as np

try:
    import convergence as cr
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TestUCLModule(unittest.TestCase):
    """Test ucl module components."""

    def test_set_heidel_welch_constants(self):
        """Test set_heidel_welch_constants function."""
        try:
            # Initialize the HeidelbergerWelch object
            heidel_welch = cr.HeidelbergerWelch()
        except CVGError:
            msg = "Failed to initialize the HeidelbergerWelch object."
            raise CVGError(msg)

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

        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=0.0)
        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            confidence_coefficient=1.0)

        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            heidel_welch_number_points=0)
        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            heidel_welch_number_points=10)

        heidel_welch.unset_heidel_welch_constants()

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

        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            heidel_welch_number_points=50.0)
        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            heidel_welch_number_points=-10.0)
        self.assertRaises(
            CVGError,
            heidel_welch.set_heidel_welch_constants,
            heidel_welch_number_points='50')

        heidel_welch.set_heidel_welch_constants()

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

        heidel_welch.set_heidel_welch_constants(heidel_welch_number_points=25)

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

    def test_ucl(self):
        """Test ucl function."""
        x = np.arange(100.)
        # x is not one dimensional
        self.assertRaises(CVGError, cr.ucl,
                          x.reshape(5, 20))
        # x does not have enough size
        self.assertRaises(CVGError, cr.ucl, x)

        x = np.ones(1000) * 10 + (np.random.random_sample(1000) - 0.5)
        _ = cr.ucl(x)
