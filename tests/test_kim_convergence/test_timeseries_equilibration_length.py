"""Test Time series equilibration_length module."""

import unittest
import numpy as np

try:
    import kim_convergence as cr
except Exception as e:  # intentional catch-all
    raise RuntimeError("Failed to import `kim-convergence` utility module") from e

from kim_convergence import CRError


class TestTimeseriesEquilibrationLengthModule(unittest.TestCase):
    """Test Time series equilibration_length module components."""

    def test_estimate_equilibration_length(self):
        """Test estimate_equilibration_length function."""
        x = np.arange(100.0)
        # si is not an str nor a valid si methods
        self.assertRaises(
            CRError, cr.estimate_equilibration_length, x, si=cr.statistical_inefficiency
        )

        self.assertRaises(CRError, cr.estimate_equilibration_length, x, si=1.0)

        self.assertRaises(CRError, cr.estimate_equilibration_length, x, si=1)

        self.assertRaises(CRError, cr.estimate_equilibration_length, x, si=True)

        self.assertRaises(CRError, cr.estimate_equilibration_length, x, si="si")

        # x is not one dimensional
        self.assertRaises(CRError, cr.estimate_equilibration_length, x.reshape(5, 20))

        # constant data sets
        x = np.ones(100)
        n, si = cr.estimate_equilibration_length(x)
        self.assertTrue(n == 0)
        self.assertTrue(si == x.size)

        rng = np.random.RandomState(12345)

        x = np.ones(100) * 10 + (rng.random_sample(100) - 0.5)

        # invalid int ignore_end
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, ignore_end=0)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, ignore_end=-1)

        # invalid float ignore_end
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, ignore_end=0.0)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, ignore_end=1.0)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, ignore_end=-0.1)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, ignore_end=1.1)

        # invalid ignore_end
        self.assertRaises(
            CRError, cr.estimate_equilibration_length, x, ignore_end="None"
        )
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, ignore_end="1")

        # invalid ignore_end
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, ignore_end=120)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, ignore_end=100)

        # insufficient data points
        n = 1
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)

        self.assertRaises(CRError, cr.estimate_equilibration_length, x)
        self.assertRaises(
            CRError,
            cr.estimate_equilibration_length,
            x,
            si="geyer_r_statistical_inefficiency",
        )
        self.assertRaises(
            CRError,
            cr.estimate_equilibration_length,
            x,
            si="geyer_split_r_statistical_inefficiency",
        )
        self.assertRaises(
            CRError,
            cr.estimate_equilibration_length,
            x,
            si="geyer_split_statistical_inefficiency",
        )

        n = 3
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)

        self.assertRaises(
            CRError,
            cr.estimate_equilibration_length,
            x,
            si="geyer_r_statistical_inefficiency",
        )

        n = 7
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)

        self.assertRaises(
            CRError,
            cr.estimate_equilibration_length,
            x,
            si="geyer_split_r_statistical_inefficiency",
        )

        self.assertRaises(
            CRError,
            cr.estimate_equilibration_length,
            x,
            si="geyer_split_statistical_inefficiency",
        )

        # invalid nskip
        n = 100
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, nskip=1.0)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, nskip=10.0)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, nskip=-10.0)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, nskip=0)
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, nskip=-1)

        rng = np.random.RandomState(12345)
        n = 1000
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        y = np.concatenate(
            (np.arange(n // 10)[::-1] / float(n // 10), np.zeros(n - n // 10))
        )

        x += y

        n1, si1 = cr.estimate_equilibration_length(x, fft=True)
        n2, si2 = cr.estimate_equilibration_length(x, fft=False)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(x, nskip=2, fft=True)
        n2, si2 = cr.estimate_equilibration_length(x, nskip=2, fft=False)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(
            x, fft=True, minimum_correlation_time=2
        )
        n2, si2 = cr.estimate_equilibration_length(
            x, fft=False, minimum_correlation_time=2
        )

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(x, fft=True, ignore_end=25)
        n2, si2 = cr.estimate_equilibration_length(x, fft=False, ignore_end=25)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(
            x, si="geyer_r_statistical_inefficiency", fft=True
        )
        n2, si2 = cr.estimate_equilibration_length(
            x, si="geyer_r_statistical_inefficiency", fft=False
        )

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(
            x, si="geyer_split_r_statistical_inefficiency", fft=True
        )
        n2, si2 = cr.estimate_equilibration_length(
            x, si="geyer_split_r_statistical_inefficiency", fft=False
        )

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(
            x, si="geyer_split_statistical_inefficiency", fft=True
        )
        n2, si2 = cr.estimate_equilibration_length(
            x, si="geyer_split_statistical_inefficiency", fft=False
        )

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        # there is at least one value in the input array
        # which is non-finite or not-number
        n = 1000
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)

        x[2] = np.inf

        self.assertRaises(CRError, cr.estimate_equilibration_length, x)

        x[2] = np.nan

        self.assertRaises(CRError, cr.estimate_equilibration_length, x)

        x[2] = -np.inf

        self.assertRaises(CRError, cr.estimate_equilibration_length, x)

    def test_invalid_solver(self):
        """An unknown solver must raise CRError."""
        rng = np.random.RandomState(12345)
        x = np.ones(100) * 10 + (rng.random_sample(100) - 0.5)

        self.assertRaisesRegex(
            CRError,
            r'invalid solver = "bogus"',
            cr.estimate_equilibration_length,
            x,
            solver="bogus",
        )
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, solver="")
        self.assertRaises(CRError, cr.estimate_equilibration_length, x, solver=None)

    def _effective_sample_size(self, x, t, si_value):
        """N_eff = (N - t) / si, the quantity the solvers maximize."""
        return (x.size - t) / si_value

    def test_solver_equivalence(self):
        """The unimodal (ternary) solver is an *approximate* maximizer: on a
        series with a decaying transient it must return a statistically
        equivalent result to the exhaustive scan -- a nearly identical
        statistical inefficiency and an effective sample size within a small
        fraction of the exhaustive optimum -- but not necessarily the identical
        integer index, because N_eff(t) is locally jagged near its (flat) peak.
        """
        rng = np.random.RandomState(12345)
        n = 1000

        # Stationary noise with a decaying transient added to the front: this
        # makes N_eff(t) rise (as the transient is removed) then plateau.
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        transient = np.concatenate(
            (np.arange(n // 10)[::-1] / float(n // 10), np.zeros(n - n // 10))
        )
        x += transient

        for fft in (True, False):
            n_exhaustive, si_exhaustive = cr.estimate_equilibration_length(
                x, fft=fft, solver="exhaustive"
            )
            n_unimodal, si_unimodal = cr.estimate_equilibration_length(
                x, fft=fft, solver="unimodal"
            )

            # Statistical inefficiency must agree closely (relative tolerance).
            self.assertAlmostEqual(
                si_unimodal, si_exhaustive, delta=0.01 * si_exhaustive + 1e-9
            )

            # The unimodal solver must not lose more than a small fraction of
            # the optimal effective sample size found by the exhaustive scan.
            neff_exhaustive = self._effective_sample_size(
                x, n_exhaustive, si_exhaustive
            )
            neff_unimodal = self._effective_sample_size(x, n_unimodal, si_unimodal)
            self.assertGreaterEqual(neff_unimodal, 0.98 * neff_exhaustive)

    def test_solver_auto_matches_exhaustive_for_small_series(self):
        """For small series, ``auto`` must behave exactly like ``exhaustive``
        (the candidate-offset count is below the fallback threshold), so the
        result is bit-for-bit identical."""
        rng = np.random.RandomState(54321)
        n = 1000
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        x += np.concatenate(
            (np.arange(n // 10)[::-1] / float(n // 10), np.zeros(n - n // 10))
        )

        n_auto, si_auto = cr.estimate_equilibration_length(x, solver="auto")
        n_exhaustive, si_exhaustive = cr.estimate_equilibration_length(
            x, solver="exhaustive"
        )

        self.assertEqual(n_auto, n_exhaustive)
        self.assertAlmostEqual(si_auto, si_exhaustive, places=12)

    def test_solver_auto_matches_unimodal_for_large_series(self):
        """For large series, ``auto`` must dispatch to the unimodal solver and
        return a bit-for-bit identical result. The candidate-offset count
        (~0.75 * n with the default ignore_end) must exceed the fallback
        threshold so that ``auto`` does not use the exhaustive scan."""
        rng = np.random.RandomState(98765)
        # n chosen so that ~0.75 * n candidate offsets exceeds the 10000
        # exhaustive-scan threshold, forcing auto -> unimodal.
        n = 20000
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        x += np.concatenate(
            (np.arange(n // 10)[::-1] / float(n // 10), np.zeros(n - n // 10))
        )

        n_auto, si_auto = cr.estimate_equilibration_length(x, solver="auto")
        n_unimodal, si_unimodal = cr.estimate_equilibration_length(
            x, solver="unimodal"
        )

        self.assertEqual(n_auto, n_unimodal)
        self.assertAlmostEqual(si_auto, si_unimodal, places=12)

    def test_unimodal_near_constant_series(self):
        """On a near-constant (already-equilibrated) series the unimodal solver
        must return a statistically equivalent result: si ~= 1 and an effective
        sample size within a small fraction of the exhaustive optimum."""
        rng = np.random.RandomState(7)
        x = np.ones(200) * 5.0 + (rng.random_sample(200) - 0.5) * 1e-3

        n_exhaustive, si_exhaustive = cr.estimate_equilibration_length(
            x, solver="exhaustive"
        )
        n_unimodal, si_unimodal = cr.estimate_equilibration_length(
            x, solver="unimodal"
        )

        self.assertAlmostEqual(
            si_unimodal, si_exhaustive, delta=0.01 * si_exhaustive + 1e-9
        )
        neff_exhaustive = self._effective_sample_size(x, n_exhaustive, si_exhaustive)
        neff_unimodal = self._effective_sample_size(x, n_unimodal, si_unimodal)
        self.assertGreaterEqual(neff_unimodal, 0.98 * neff_exhaustive)
