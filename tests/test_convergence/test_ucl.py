"""Test ucl module."""
import unittest
import numpy as np

try:
    import convergence as cvg
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TestUCLModule(unittest.TestCase):
    """Test ucl module components."""

    def test_set_heidel_welch_constants(self):
        """Test set_heidel_welch_constants function."""
        heidel_welch_set, \
            heidel_welch_k, \
            heidel_welch_n, \
            heidel_welch_p, \
            A, \
            Aplus_1, \
            Aplus_2, \
            Aplus_3, \
            heidel_welch_C1_1, \
            heidel_welch_C1_2, \
            heidel_welch_C1_3, \
            heidel_welch_C2_1, \
            heidel_welch_C2_2, \
            heidel_welch_C2_3, \
            tm_1, \
            tm_2, \
            tm_3 = cvg.get_heidel_welch_constants()

        self.assertFalse(heidel_welch_set)
        self.assertIsNone(heidel_welch_k)
        self.assertIsNone(heidel_welch_n)
        self.assertIsNone(heidel_welch_p)
        self.assertIsNone(A)
        self.assertIsNone(Aplus_1)
        self.assertIsNone(Aplus_2)
        self.assertIsNone(Aplus_3)
        self.assertIsNone(heidel_welch_C1_1)
        self.assertIsNone(heidel_welch_C1_2)
        self.assertIsNone(heidel_welch_C1_3)
        self.assertIsNone(heidel_welch_C2_1)
        self.assertIsNone(heidel_welch_C2_2)
        self.assertIsNone(heidel_welch_C2_3)
        self.assertIsNone(tm_1)
        self.assertIsNone(tm_2)
        self.assertIsNone(tm_3)

        self.assertRaises(CVGError, cvg.set_heidel_welch_constants, p=0.0)
        self.assertRaises(CVGError, cvg.set_heidel_welch_constants, p=1.0)
        
        self.assertRaises(CVGError, cvg.set_heidel_welch_constants, k=0)
        self.assertRaises(CVGError, cvg.set_heidel_welch_constants, k=10)
        self.assertRaises(CVGError, cvg.set_heidel_welch_constants, k=50.0)
        self.assertRaises(CVGError, cvg.set_heidel_welch_constants, k=-10.0)
        self.assertRaises(CVGError, cvg.set_heidel_welch_constants, k='50')

        cvg.set_heidel_welch_constants()

        heidel_welch_set = cvg.get_heidel_welch_set()

        self.assertTrue(heidel_welch_set)

        heidel_welch_k, heidel_welch_n, \
            heidel_welch_p = cvg.get_heidel_welch_knp()

        self.assertTrue(heidel_welch_k == 50)
        self.assertTrue(heidel_welch_n == 200)
        self.assertTrue(heidel_welch_p == 0.975)

        heidel_welch_C1_1, heidel_welch_C1_2, \
            heidel_welch_C1_3 = cvg.get_heidel_welch_C1()

        self.assertAlmostEqual(heidel_welch_C1_1, 0.974, places=3)
        self.assertAlmostEqual(heidel_welch_C1_2, 0.941, places=3)
        # In the paper it is rounded up to 0.895
        self.assertAlmostEqual(heidel_welch_C1_3, 0.894, places=3)

        heidel_welch_C2_1, heidel_welch_C2_2, \
            heidel_welch_C2_3 = cvg.get_heidel_welch_C2()

        self.assertTrue(heidel_welch_C2_1 == 37)
        self.assertTrue(heidel_welch_C2_2 == 16)
        self.assertTrue(heidel_welch_C2_3 == 8)

        cvg.set_heidel_welch_constants(k=25)

        heidel_welch_k, heidel_welch_n, \
            heidel_welch_p = cvg.get_heidel_welch_knp()

        self.assertTrue(heidel_welch_k == 25)
        self.assertTrue(heidel_welch_n == 100)
        self.assertTrue(heidel_welch_p == 0.975)

        heidel_welch_C1_1, heidel_welch_C1_2, \
            heidel_welch_C1_3 = cvg.get_heidel_welch_C1()

        self.assertAlmostEqual(heidel_welch_C1_1, 0.948, places=3)
        # In the paper it is rounded up to 0.882
        self.assertAlmostEqual(heidel_welch_C1_2, 0.881, places=3)
        # In the paper it is rounded up to 0.784
        self.assertAlmostEqual(heidel_welch_C1_3, 0.783, places=3)

        heidel_welch_C2_1, heidel_welch_C2_2, \
            heidel_welch_C2_3 = cvg.get_heidel_welch_C2()

        self.assertTrue(heidel_welch_C2_1 == 18)
        self.assertTrue(heidel_welch_C2_2 == 7)
        self.assertTrue(heidel_welch_C2_3 == 3)

    def test_ucl(self):
        """Test ucl function."""
        x = np.arange(100.)
        # x is not one dimensional
        self.assertRaises(CVGError, cvg.ucl,
                          x.reshape(5, 20))
        # x does not have enough size
        self.assertRaises(CVGError, cvg.ucl, x)

        x = np.ones(1000) * 10 + (np.random.random_sample(1000) - 0.5)
        _ = cvg.ucl(x)
