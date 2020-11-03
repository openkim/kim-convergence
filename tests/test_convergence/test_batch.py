"""Test batch module."""
import unittest
import numpy as np

try:
    import convergence as cvg
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TestBatchModule(unittest.TestCase):
    """Test batch module components."""

    def test_batch(self):
        """Test batch function."""
        x = np.ones(10)

        # dimension
        self.assertRaises(CVGError, cvg.batch, x.reshape((2, 5)))

        # batch_size
        self.assertRaises(CVGError, cvg.batch, x, batch_size=5.0)
        self.assertRaises(CVGError, cvg.batch, x, batch_size=-1)
        self.assertRaises(CVGError, cvg.batch, x, batch_size=20)
        self.assertRaises(CVGError, cvg.batch, x, batch_size='20')
        self.assertRaises(CVGError, cvg.batch, x, batch_size=None)

        # scale method
        self.assertRaises(CVGError, cvg.batch, x,
                          batch_size=5, scale=cvg.translate_scale)
        self.assertRaises(CVGError, cvg.batch, x,
                          batch_size=5, scale='new_scale_metod')
        self.assertRaises(CVGError, cvg.batch, x,
                          batch_size=5, scale=None)

        # n_batches
        n = 100
        x = np.ones(n) * 10 + (np.random.random_sample(n) - 0.5)
        self.assertRaises(CVGError, cvg.batch, x, batch_size=101)

        x = np.ones(n)
        b = cvg.batch(x,
                      batch_size=5,
                      with_centering=False,
                      with_scaling=False)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cvg.batch(x,
                      batch_size=5,
                      with_centering=False,
                      with_scaling=True)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cvg.batch(x,
                      batch_size=5,
                      with_centering=True,
                      with_scaling=True)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cvg.batch(x,
                      batch_size=5,
                      with_centering=True,
                      with_scaling=False)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cvg.batch(x,
                      batch_size=7,
                      with_centering=False,
                      with_scaling=False)
        for i in b:
            self.assertTrue(i == 1.0)
        b = cvg.batch(x,
                      batch_size=7,
                      with_centering=False,
                      with_scaling=True)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cvg.batch(x,
                      batch_size=7,
                      with_centering=True,
                      with_scaling=True)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cvg.batch(x,
                      batch_size=7,
                      with_centering=True,
                      with_scaling=False)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cvg.batch(x,
                      batch_size=5,
                      scale='translate_scale',
                      with_centering=False,
                      with_scaling=False)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cvg.batch(x,
                      batch_size=5,
                      scale='translate_scale',
                      with_centering=False,
                      with_scaling=True)
        for i in b:
            self.assertTrue(i == 1.0)

        b = cvg.batch(x,
                      batch_size=5,
                      scale='translate_scale',
                      with_centering=True,
                      with_scaling=True)
        for i in b:
            self.assertTrue(i == 0.0)

        b = cvg.batch(x,
                      batch_size=5,
                      scale='translate_scale',
                      with_centering=True,
                      with_scaling=False)
        for i in b:
            self.assertTrue(i == 0.0)
