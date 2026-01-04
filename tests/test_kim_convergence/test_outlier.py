"""Test outlier module."""

import unittest
import numpy as np

try:
    import kim_convergence as cr
except Exception as e:  # intentional catch-all
    raise RuntimeError("Failed to import `kim-convergence` utility module") from e


class TestOutlierModule(unittest.TestCase):
    """Test outlier module components."""

    def test_outlier_test(self):
        """Test outlier_test function."""
        x = np.array([56, 43, 51, 47, 100, 76, 56, 53, 49, 159, 13, 73, 50, 24,
                      39, 86, 313, 42, 1, 6, 21, 48, 27, 78, 98, 60, 80, 24,
                      91, 19, 81, 47, 43])

        outlier_indices = cr.outlier_test(x, outlier_method="iqr")
        outlier_x = x[outlier_indices]
        self.assertIsInstance(outlier_indices, np.ndarray)
        assert isinstance(outlier_indices, np.ndarray)  # keeps mypy happy
        self.assertTrue(outlier_indices.size == 2)
        self.assertTrue(outlier_x[0] == 159)
        self.assertTrue(outlier_x[1] == 313)

        outlier_indices = cr.outlier_test(x, outlier_method="extreme_iqr")
        outlier_x = x[outlier_indices]
        self.assertIsInstance(outlier_indices, np.ndarray)
        assert isinstance(outlier_indices, np.ndarray)  # keeps mypy happy
        self.assertTrue(outlier_indices.size == 1)
        self.assertTrue(outlier_x[0] == 313)

        outlier_indices = cr.outlier_test(x, outlier_method="z_score")
        outlier_x = x[outlier_indices]
        self.assertIsInstance(outlier_indices, np.ndarray)
        assert isinstance(outlier_indices, np.ndarray)  # keeps mypy happy
        self.assertTrue(outlier_indices.size == 1)
        self.assertTrue(outlier_x[0] == 313)

        outlier_indices = cr.outlier_test(x, outlier_method="modified_z_score")
        outlier_x = x[outlier_indices]
        self.assertIsInstance(outlier_indices, np.ndarray)
        assert isinstance(outlier_indices, np.ndarray)  # keeps mypy happy
        self.assertTrue(outlier_indices.size == 1)
        self.assertTrue(outlier_x[0] == 313)
