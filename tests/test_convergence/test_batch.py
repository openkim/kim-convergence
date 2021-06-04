"""Test batch module."""
import unittest
import numpy as np

try:
    import convergence as cr
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TestBatchModule(unittest.TestCase):
    """Test batch module components."""

    def test_batch(self):
        """Test batch function."""
        x = np.ones(10)

        # dimension
        self.assertRaises(CVGError, cr.batch, x.reshape((2, 5)))

        # batch_size
        self.assertRaises(CVGError, cr.batch, x, batch_size=5.0)
        self.assertRaises(CVGError, cr.batch, x, batch_size=-1)
        self.assertRaises(CVGError, cr.batch, x, batch_size=20)
        self.assertRaises(CVGError, cr.batch, x, batch_size='20')
        self.assertRaises(CVGError, cr.batch, x, batch_size=None)

        # scale method
        self.assertRaises(CVGError, cr.batch, x,
                          batch_size=5, scale=cr.translate_scale,
                          with_centering=True)
        self.assertRaises(CVGError, cr.batch, x,
                          batch_size=5, scale='new_scale_metod',
                          with_centering=True)
        self.assertRaises(CVGError, cr.batch, x,
                          batch_size=5, scale=None,
                          with_centering=True)

        # n_batches
        rng = np.random.RandomState(12345)

        n = 100
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        self.assertRaises(CVGError, cr.batch, x, batch_size=101)

        x = np.ones(n)
        b = cr.batch(x,
                     batch_size=5,
                     with_centering=False,
                     with_scaling=False)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cr.batch(x,
                     batch_size=5,
                     with_centering=False,
                     with_scaling=True)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cr.batch(x,
                     batch_size=5,
                     with_centering=True,
                     with_scaling=True)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cr.batch(x,
                     batch_size=5,
                     with_centering=True,
                     with_scaling=False)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cr.batch(x,
                     batch_size=7,
                     with_centering=False,
                     with_scaling=False)
        for i in b:
            self.assertTrue(i == 1.0)
        b = cr.batch(x,
                     batch_size=7,
                     with_centering=False,
                     with_scaling=True)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cr.batch(x,
                     batch_size=7,
                     with_centering=True,
                     with_scaling=True)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cr.batch(x,
                     batch_size=7,
                     with_centering=True,
                     with_scaling=False)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cr.batch(x,
                     batch_size=5,
                     scale='translate_scale',
                     with_centering=False,
                     with_scaling=False)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cr.batch(x,
                     batch_size=5,
                     scale='translate_scale',
                     with_centering=False,
                     with_scaling=True)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cr.batch(x,
                     batch_size=5,
                     scale='translate_scale',
                     with_centering=True,
                     with_scaling=True)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cr.batch(x,
                     batch_size=5,
                     scale='translate_scale',
                     with_centering=True,
                     with_scaling=False)
        for i in b:
            self.assertTrue(i == 0.0)

        x[0] = np.nan
        self.assertRaises(CVGError, cr.batch, x)

        x[0] = np.inf
        self.assertRaises(CVGError, cr.batch, x)

        x[0] = np.NaN
        self.assertRaises(CVGError, cr.batch, x)

        rng = np.random.RandomState(12345)

        n = 1000
        x = np.ones(n) * 100 + (rng.random_sample(n) - 0.5)

        b = cr.batch(x)

        self.assertEqual(b.size, n // 5)
        self.assertAlmostEqual(b.mean(), 100.0, places=2)

        # test median as a reduction function

        test_passed = True

        try:
            b = cr.batch(x, func=np.median)
        except CVGError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertEqual(b.size, n // 5)
        self.assertAlmostEqual(b.mean(), 100.0, places=1)
