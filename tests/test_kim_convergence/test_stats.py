"""Test stats module."""

import unittest
import numpy as np

try:
    import kim_convergence as cr
except Exception as e:  # intentional catch-all
    raise RuntimeError("Failed to import `kim-convergence` utility module") from e

from kim_convergence import CRError


class TestStatsModule(unittest.TestCase):
    """Test stats module components."""

    def test_get_fft_optimal_size(self):
        """Test get_fft_optimal_size function."""
        self.assertTrue(cr.get_fft_optimal_size(1) == 1)
        self.assertTrue(cr.get_fft_optimal_size(6) == 6)
        self.assertTrue(cr.get_fft_optimal_size(7) == 8)
        self.assertTrue(cr.get_fft_optimal_size(8) == 8)
        self.assertTrue(cr.get_fft_optimal_size(9) == 9)
        self.assertTrue(cr.get_fft_optimal_size(10) == 10)
        self.assertTrue(cr.get_fft_optimal_size(11) == 12)

        for i in range(10):
            n2 = 2**i
            self.assertTrue(cr.get_fft_optimal_size(n2) == n2)

            n3 = 3**i
            self.assertTrue(cr.get_fft_optimal_size(n3) == n3)

            n5 = 5**i
            self.assertTrue(cr.get_fft_optimal_size(n5) == n5)

            n23 = n2 * n3
            self.assertTrue(cr.get_fft_optimal_size(n23) == n23)

            n25 = n2 * n5
            self.assertTrue(cr.get_fft_optimal_size(n25) == n25)

            n35 = n3 * n5
            self.assertTrue(cr.get_fft_optimal_size(n35) == n35)

            n235 = n2 * n3 * n5
            self.assertTrue(n235 == cr.get_fft_optimal_size(n235))

        self.assertTrue(cr.get_fft_optimal_size(162) == 162)

        for i in range(163, 180):
            self.assertTrue(cr.get_fft_optimal_size(i) == 180)

        self.assertTrue(cr.get_fft_optimal_size(6144) == 6144)

        for i in range(6145, 6250):
            self.assertTrue(cr.get_fft_optimal_size(i) == 6250)

        self.assertTrue(cr.get_fft_optimal_size(90000) == 90000)

        for i in range(90001, 91125):
            self.assertTrue(cr.get_fft_optimal_size(i) == 91125)

        for i in range(98416, 100000):
            self.assertTrue(cr.get_fft_optimal_size(i) == 100000)

        self.assertTrue(cr.get_fft_optimal_size(100000) == 100000)

    def test_auto_covariance(self):
        """Test auto_covariance function."""
        a0 = np.array(
            [
                8.25,
                5.775,
                3.399999999999999,
                1.225,
                -0.65,
                -2.124999999999999,
                -3.1,
                -3.475,
                -3.15,
                -2.025,
            ],
            dtype=np.float64,
        )

        a = np.arange(1, 11.0)

        test_passed = True

        try:
            np.testing.assert_allclose(
                a0, cr.auto_covariance(a), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            np.testing.assert_allclose(
                a0, cr.auto_covariance(a, fft=True), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            np.testing.assert_allclose(
                cr.auto_covariance(a, fft=False),
                cr.auto_covariance(a, fft=True),
                rtol=1e-14,
                atol=1e-14,
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a = np.arange(-11.0, 11.0)

        try:
            np.testing.assert_allclose(
                cr.auto_covariance(a, fft=False),
                cr.auto_covariance(a, fft=True),
                rtol=1e-14,
                atol=1e-14,
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a = np.random.rand(1000)

        try:
            np.testing.assert_allclose(
                cr.auto_covariance(a, fft=False),
                cr.auto_covariance(a, fft=True),
                rtol=1e-14,
                atol=1e-14,
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a = np.random.rand(2, 3)

        self.assertRaises(CRError, cr.auto_covariance, a)

        a = []

        self.assertRaises(CRError, cr.auto_covariance, a)

        a = np.random.rand(1000)
        a[100] = np.inf

        self.assertRaises(CRError, cr.auto_covariance, a)

        a[100] = np.nan

        self.assertRaises(CRError, cr.auto_covariance, a)

        a[100] = -np.inf

        self.assertRaises(CRError, cr.auto_covariance, a)

    def test_cross_covariance(self):
        """Test cross_covariance function."""
        a0 = np.array(
            [
                8.25,
                5.775,
                3.399999999999999,
                1.225,
                -0.65,
                -2.124999999999999,
                -3.1,
                -3.475,
                -3.15,
                -2.025,
            ],
            dtype=np.float64,
        )

        a = np.arange(1, 11.0)

        test_passed = True

        try:
            np.testing.assert_allclose(
                a0, cr.cross_covariance(a, None), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            np.testing.assert_allclose(
                a0, cr.cross_covariance(a, a), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a = np.random.rand(2, 3)

        self.assertRaises(CRError, cr.cross_covariance, a, a)

        a = np.random.rand(2, 3)
        b = np.random.rand(2)

        self.assertRaises(CRError, cr.cross_covariance, a, b)

        a = np.random.rand(3)
        b = np.random.rand(2)

        self.assertRaises(CRError, cr.cross_covariance, a, b)

        a = np.random.rand(3, 2)
        b = np.random.rand(2)

        self.assertRaises(CRError, cr.cross_covariance, a, b)

        a = []
        b = []

        self.assertRaises(CRError, cr.cross_covariance, a, b)

        ab = np.array(
            [
                -4.783685406261340e-03,
                -1.138527201306546e-02,
                -5.809494077151546e-03,
                -7.156510315177644e-03,
                -5.585275066021979e-03,
                9.824294199818826e-05,
                1.343198013572316e-02,
                1.093106704175784e-03,
                3.212505155860149e-04,
                4.031577169803108e-03,
            ],
            dtype=np.float64,
        )

        a = np.array(
            [
                0.274009201055018,
                0.842734711117661,
                0.265924933120814,
                0.326235322789641,
                0.737824574252726,
                0.414427991215862,
                0.095971984892582,
                0.352737912273902,
                0.457335906534869,
                0.287764433084643,
            ],
            dtype=np.float64,
        )

        b = np.array(
            [
                0.120721802214958,
                0.285091802721577,
                0.44536018074963,
                0.294508675395237,
                0.830698229189781,
                0.241645865988203,
                0.978724978429237,
                0.779406362577084,
                0.423941089234571,
                0.231479558899855,
            ],
            dtype=np.float64,
        )

        try:
            np.testing.assert_allclose(
                ab, cr.cross_covariance(a, b), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a = np.random.rand(1000)
        b = np.random.rand(1000)

        try:
            np.testing.assert_allclose(
                cr.cross_covariance(a, b, fft=False),
                cr.cross_covariance(a, b, fft=True),
                rtol=1e-14,
                atol=1e-14,
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a[100] = np.inf

        self.assertRaises(CRError, cr.cross_covariance, a, b)

        a[100] = np.nan

        self.assertRaises(CRError, cr.cross_covariance, a, b)

        a[100] = -np.inf

        self.assertRaises(CRError, cr.cross_covariance, a, b)

        a[100] = np.random.rand()
        b[101] = np.inf

        self.assertRaises(CRError, cr.cross_covariance, a, b)

        b[101] = np.nan

        self.assertRaises(CRError, cr.cross_covariance, a, b)

        a[101] = -np.inf

        self.assertRaises(CRError, cr.cross_covariance, a, b)

    def test_auto_correlate(self):
        """Test auto_correlate function."""
        a0 = np.array(
            [
                1.0,
                0.7,
                0.412121212121212,
                0.148484848484848,
                -0.078787878787879,
                -0.257575757575758,
                -0.375757575757576,
                -0.421212121212121,
                -0.381818181818182,
                -0.245454545454545,
            ],
            dtype=np.float64,
        )

        a = np.arange(1, 11.0)

        test_passed = True

        try:
            np.testing.assert_allclose(
                a0, cr.auto_correlate(a, nlags=a.size + 1), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            np.testing.assert_allclose(a0, cr.auto_correlate(a), rtol=1e-14, atol=1e-14)
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        for lags in range(a.size - 2, 1, -1):
            try:
                np.testing.assert_allclose(
                    a0[: lags + 1],
                    cr.auto_correlate(a, nlags=lags),
                    rtol=1e-14,
                    atol=1e-14,
                )
            except AssertionError:
                test_passed = False

            self.assertTrue(test_passed)

        self.assertRaises(CRError, cr.auto_correlate, a, nlags=1.0)

        self.assertRaises(CRError, cr.auto_correlate, a, nlags=0)

        a = np.arange(-11.0, 11.0)

        try:
            np.testing.assert_allclose(
                cr.auto_correlate(a, fft=False),
                cr.auto_correlate(a, fft=True),
                rtol=1e-14,
                atol=1e-14,
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a = np.random.rand(1000)

        try:
            np.testing.assert_allclose(
                cr.auto_correlate(a, fft=False),
                cr.auto_correlate(a, fft=True),
                rtol=1e-14,
                atol=1e-14,
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a = np.ones(10)

        self.assertRaises(CRError, cr.auto_correlate, a)

        a = np.random.rand(2, 3)

        self.assertRaises(CRError, cr.auto_correlate, a)

        a = []

        self.assertRaises(CRError, cr.auto_correlate, a)

        a = np.random.rand(1800)
        a[10] = np.inf

        self.assertRaises(CRError, cr.auto_correlate, a)

        a[10] = np.nan

        self.assertRaises(CRError, cr.auto_correlate, a)

        a[10] = -np.inf

        self.assertRaises(CRError, cr.auto_correlate, a)

    def test_cross_correlate(self):
        """Test cross_correlate function."""
        a0 = np.array(
            [
                1.0,
                0.7,
                0.412121212121212,
                0.148484848484848,
                -0.078787878787879,
                -0.257575757575758,
                -0.375757575757576,
                -0.421212121212121,
                -0.381818181818182,
                -0.245454545454545,
            ],
            dtype=np.float64,
        )

        a = np.arange(1, 11.0)

        test_passed = True

        try:
            np.testing.assert_allclose(
                a0, cr.cross_correlate(a, None), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        try:
            np.testing.assert_allclose(
                a0, cr.cross_correlate(a, a), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a = np.random.rand(2, 3)

        self.assertRaises(CRError, cr.cross_correlate, a, a)

        a = np.random.rand(2, 3)
        b = np.random.rand(2)

        self.assertRaises(CRError, cr.cross_correlate, a, b)

        a = np.random.rand(3)
        b = np.random.rand(2)

        self.assertRaises(CRError, cr.cross_correlate, a, b)

        a = np.random.rand(3, 2)
        b = np.random.rand(2)

        self.assertRaises(CRError, cr.cross_correlate, a, b)

        a = []
        b = []

        self.assertRaises(CRError, cr.cross_correlate, a, b)

        ab = np.array(
            [
                0.042606000505134,
                -0.329786944440556,
                0.192637392475732,
                0.155673928115373,
                0.522912208147887,
                0.164804634946593,
                0.188353306047153,
                -0.088233151288948,
                0.12911468123748,
                -0.179572682320018,
                -0.042809526704892,
                -0.221866360790168,
                -0.077043562380959,
                0.045515405712984,
                0.086871223237283,
            ],
            dtype=np.float64,
        )

        a = np.array(
            [
                0.055507369988729,
                0.481275530315665,
                0.576213089607893,
                0.016740675908713,
                0.105691668871077,
                0.024514304748295,
                0.757165657685533,
                0.94289533012766,
                0.312900838839897,
                0.969586615778014,
                0.302377996207441,
                0.814324986544453,
                0.569858334526404,
                0.232845471699136,
                0.084149261439085,
            ],
            dtype=np.float64,
        )

        b = np.array(
            [
                0.081476060112686,
                0.493472747535988,
                0.645983306825734,
                0.940062256070189,
                0.653280702536729,
                0.952501391577221,
                0.341948751974199,
                0.866109187996785,
                0.160089305528428,
                0.63450107042816,
                0.330938100271903,
                0.039784917727305,
                0.765640137311665,
                0.113114387305061,
                0.211818397684178,
            ],
            dtype=np.float64,
        )

        try:
            np.testing.assert_allclose(
                ab, cr.cross_correlate(a, b), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        for lags in range(a.size - 2, 1, -1):
            try:
                np.testing.assert_allclose(
                    ab[: lags + 1],
                    cr.cross_correlate(a, b, nlags=lags),
                    rtol=1e-14,
                    atol=1e-14,
                )
            except AssertionError:
                test_passed = False

            self.assertTrue(test_passed)

        self.assertRaises(CRError, cr.cross_correlate, a, b, nlags=1.0)

        self.assertRaises(CRError, cr.cross_correlate, a, b, nlags=0)

        a = np.random.rand(1200)
        b = np.random.rand(1200)

        try:
            np.testing.assert_allclose(
                cr.cross_correlate(a, b, fft=False),
                cr.cross_correlate(a, b, fft=True),
                rtol=1e-14,
                atol=1e-14,
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        a[100] = np.inf

        self.assertRaises(CRError, cr.cross_correlate, a, b)

        a[1000] = np.nan

        self.assertRaises(CRError, cr.cross_correlate, a, b)

        a[1000] = -np.inf

        self.assertRaises(CRError, cr.cross_correlate, a, b)

        a[1000] = np.random.rand()
        b[1001] = np.inf

        self.assertRaises(CRError, cr.cross_correlate, a, b)

        b[1001] = np.nan

        self.assertRaises(CRError, cr.cross_correlate, a, b)

        a[1001] = -np.inf

        self.assertRaises(CRError, cr.cross_correlate, a, b)

    def test_periodogram(self):
        """Test periodogram function."""
        x = np.arange(20.0).reshape(4, 5)

        self.assertRaises(CRError, cr.periodogram, x)

        x = np.array(
            [
                -0.026194534693644133672,
                -0.0087625583491129695884,
                -0.040715708436513946278,
                -0.0055650914136230354365,
                0.0022293404175303859274,
                -0.0079331491517730823998,
                -0.026851666641645354633,
                -0.010494064882079965489,
                -0.0063984847377457788192,
                0.00022605290844075371661,
                -0.036614515756249028933,
                0.0048156981782889376337,
                0.024418093109002423496,
                -0.029591063698982272151,
                0.035182584059791639775,
                0.0072580743504497670718,
                -0.033679775536749473330,
                -0.038814256707319666484,
                0.012495104678473908932,
                0.0061158018632598287745,
                0.025063766801558445829,
            ],
            dtype=np.float64,
        )

        _pdgm = np.array(
            [
                0.000019041632561446798652,
                0.0000045945687416506509257,
                0.000072448134042988333024,
                0.000091673981633281091176,
                0.000013475072829033355698,
                0.000094429340951708570849,
                0.000036881053403977916941,
                0.000059812715719436275225,
                0.0000017483195031516726558,
                0.000075130150597221359858,
            ],
            dtype=np.float64,
        )

        pdgm = cr.periodogram(x)

        for p, _p in zip(pdgm, _pdgm):
            self.assertAlmostEqual(p, _p, places=12)

        pdgm = cr.periodogram(x, fft=True)

        for p, _p in zip(pdgm, _pdgm):
            self.assertAlmostEqual(p, _p, places=12)

        x = np.arange(1.0, 21.0, 0.5)
        _pdgm = cr.periodogram(x, fft=True)
        pdgm = cr.periodogram(x, fft=False)

        for p, _p in zip(pdgm, _pdgm):
            self.assertAlmostEqual(p, _p, places=12)

        x = np.arange(1.0, 21.0, 0.1)
        _pdgm = cr.periodogram(x, fft=True, with_mean=True)
        pdgm = cr.periodogram(x, fft=False, with_mean=True)

        for p, _p in zip(pdgm, _pdgm):
            self.assertAlmostEqual(p, _p, places=12)

        x = np.arange(1.0, 21.0, 0.5)
        x[2] = np.inf

        self.assertRaises(CRError, cr.periodogram, x)

        x[2] = np.nan

        self.assertRaises(CRError, cr.periodogram, x)

        x[2] = -np.inf

        self.assertRaises(CRError, cr.periodogram, x)

    def test_int_power(self):
        """Test int_power function."""
        x = np.arange(6)

        test_passed = True

        try:
            np.testing.assert_allclose(
                np.power(x, 4), cr.int_power(x, 4), rtol=1e-14, atol=1e-14
            )
        except AssertionError:
            test_passed = False

        self.assertTrue(test_passed)

        self.assertRaises(CRError, cr.int_power, [np.inf], 2)
        self.assertRaises(CRError, cr.int_power, 2, 2)
        self.assertRaises(CRError, cr.int_power, x.reshape((2, 3)), 2)
        self.assertRaises(CRError, cr.int_power, x, 2.1)

    def test_moment(self):
        """Test moment function."""
        x = np.arange(6)
        self.assertRaises(CRError, cr.moment, x.reshape((2, 3)), moment=2)
        self.assertRaises(CRError, cr.moment, 2, moment=2)
        self.assertRaises(CRError, cr.moment, [np.inf], moment=2)
        self.assertRaises(CRError, cr.moment, x, moment=2.1)

        self.assertEqual(cr.moment(x, moment=1), 0)

        x = np.arange(1, 6)
        self.assertEqual(cr.moment(x, moment=2), 2.0)

    def test_skew(self):
        """Test skew function."""
        x = np.arange(1, 6)
        self.assertEqual(cr.skew(x), 0.0)

        x = np.array([2, 8, 0, 4, 1, 9, 9, 0])
        self.assertEqual(cr.skew(x, bias=True), 0.2650554122698573)

        x = np.array([2])
        self.assertEqual(cr.skew(x, bias=True), 0.0)

        x = np.array([2, 2])
        self.assertEqual(cr.skew(x, bias=True), 0.0)
