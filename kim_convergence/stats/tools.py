"""Tools module.

Helper functions for time series analysis.
"""

from bisect import bisect_left
from math import isclose, pi, sqrt
import numpy as np
from typing import Optional, Union, List

from kim_convergence._default import _DEFAULT_ABS_TOL
from kim_convergence import CRError

__all__ = [
    'get_fft_optimal_size',
    'auto_covariance',
    'auto_correlate',
    'cross_covariance',
    'cross_correlate',
    'modified_periodogram',
    'periodogram',
    'int_power',
    'moment',
    'skew',
]


FFTURN = (8, 9, 10, 12, 15, 16, 18, 20,
          24, 25, 27, 30, 32, 36, 40, 45,
          48, 50, 54, 60, 64, 72, 75, 80,
          81, 90, 96, 100, 108, 120, 125, 128,
          135, 144, 150, 160, 162, 180, 192, 200,
          216, 225, 240, 243, 250, 256, 270, 288,
          300, 320, 324, 360, 375, 384, 400, 405,
          432, 450, 480, 486, 500, 512, 540, 576,
          600, 625, 640, 648, 675, 720, 729, 750,
          768, 800, 810, 864, 900, 960, 972, 1000,
          1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280,
          1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620,
          1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048,
          2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560,
          2592, 2700, 2880, 2916, 3000, 3072, 3125, 3200,
          3240, 3375, 3456, 3600, 3645, 3750, 3840, 3888,
          4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800,
          4860, 5000, 5120, 5184, 5400, 5625, 5760, 5832,
          6000, 6075, 6144, 6250, 6400, 6480, 6561, 6750,
          6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
          8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720,
          10000, 10125, 10240, 10368, 10800, 10935, 11250, 11520,
          11664, 12000, 12150, 12288, 12500, 12800, 12960, 13122,
          13500, 13824, 14400, 14580, 15000, 15360, 15552, 15625,
          16000, 16200, 16384, 16875, 17280, 17496, 18000, 18225,
          18432, 18750, 19200, 19440, 19683, 20000, 20250, 20480,
          20736, 21600, 21870, 22500, 23040, 23328, 24000, 24300,
          24576, 25000, 25600, 25920, 26244, 27000, 27648, 28125,
          28800, 29160, 30000, 30375, 30720, 31104, 31250, 32000,
          32400, 32768, 32805, 33750, 34560, 34992, 36000, 36450,
          36864, 37500, 38400, 38880, 39366, 40000, 40500, 40960,
          41472, 43200, 43740, 45000, 46080, 46656, 46875, 48000,
          48600, 49152, 50000, 50625, 51200, 51840, 52488, 54000,
          54675, 55296, 56250, 57600, 58320, 59049, 60000, 60750,
          61440, 62208, 62500, 64000, 64800, 65536, 65610, 67500,
          69120, 69984, 72000, 72900, 73728, 75000, 76800, 77760,
          78125, 78732, 80000, 81000, 81920, 82944, 84375, 86400,
          87480, 90000, 91125, 92160, 93312, 93750, 96000, 97200,
          98304, 98415, 100000)
"""tuple: FFT unique regular numbers."""


def get_fft_optimal_size(input_size: int) -> int:
    """Find the optimal size for the FFT solver.

    Get the next regular number greater than or equal to input_size [1]_.
    Regular numbers are composites of the prime factors 2, 3, and 5. Also
    known as 5-smooth numbers or Hamming numbers, these are the optimal size
    for inputs to FFT solvers.

    Args:
        input_size (int): Input data size we want to use the FFT solver on
            it. This is the length to start searching from it and is a
            positive integer.

    Returns:
        int: The first 5-smooth number greater than or equal to `input_size`.

    References:
        .. [1] Statsmodels: statistical modeling and econometrics in Python
               http://www.statsmodels.org

    """
    # Check inputs
    if not isinstance(input_size, int):
        msg = 'input_size must be an `int`.'
        raise CRError(msg)

    if input_size < 7:
        if input_size < 0:
            msg = 'input_size must be a positive `int`.'
            raise CRError(msg)
        return input_size

    # Return if it is power of 2
    if not input_size & (input_size - 1):
        return input_size

    # Get result quickly.
    if input_size < 100001:
        return FFTURN[bisect_left(FFTURN, input_size)]

    optimal_size = float('inf')
    power_5 = 1

    while power_5 < input_size:
        power_3_5 = power_5

        while power_3_5 < input_size:
            # Ceiling integer division, avoiding conversion to float
            quotient = -(-input_size // power_3_5)
            power_2 = 2 ** ((quotient - 1).bit_length())

            power_2_3_5 = power_2 * power_3_5

            if power_2_3_5 == input_size:
                return power_2_3_5

            if power_2_3_5 < optimal_size:
                optimal_size = power_2_3_5

            power_3_5 *= 3
            if power_3_5 == input_size:
                return power_3_5

        if power_3_5 < optimal_size:
            optimal_size = power_3_5

        power_5 *= 5
        if power_5 == input_size:
            return power_5

    if power_5 < optimal_size:
        optimal_size = power_5

    return optimal_size


def auto_covariance(x: Union[np.ndarray, List[float]],
                    *,
                    fft: bool = False) -> np.ndarray:
    """Calculate biased auto-covariance estimates.

    Compute auto-covariance estimates for every lag for the input array.
    This estimator is biased.

    .. math::

        \gamma_k = \frac{1}{N}\sum\limits_{t=1}^{N-K}(x_t-\Bar{x})(x_{t+K}-\Bar{x})


    Note:
        Some sources use the following formula for computing the
        autocovariance:

        .. math::

            \gamma_k = \frac{1}{N-K}\sum\limits_{t=1}^{N-K}(x_t-\Bar{x})(x_{t+K}-\Bar{x})

        This definition has less bias, than the one used here. But the
        :math:`\frac{1}{N}` formulation has some desirable statistical
        properties and is the most commonly used in the statistics literature.

    Args:
        x (array_like, 1d): Time series data.
        fft (bool, optional): If True, use FFT convolution. FFT should be
            preferred for long time series. (default: False)

    Returns:
        1darray: The estimated autocovariances.

    """
    x = np.asarray(x)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CRError(msg)

    # Data size
    x_size = x.size

    if x_size < 1:
        msg = 'x is empty.'
        raise CRError(msg)

    if not np.all(np.isfinite(x)):
        msg = 'there is at least one value in the input array which is '
        msg += 'non-finite or not-number.'
        raise CRError(msg)

    # Fluctuations
    dx = x - np.mean(x)

    if fft:
        # Find the optimal size for the FFT solver
        optimal_size = get_fft_optimal_size(2 * x_size)

        # Compute the one-dimensional discrete Fourier Transform
        dft = np.fft.rfft(dx, n=optimal_size)
        dft *= np.conjugate(dft)

        # Compute the one-dimensional inverse discrete Fourier Transform
        autocov = np.fft.irfft(dft, n=optimal_size)[:x_size]

        # Get the real part
        autocov = autocov.real
    else:
        # Auto correlation of a one-dimensional sequence
        autocov = np.correlate(dx, dx, 'full')[x_size - 1:]

    autocov /= float(x_size)

    return autocov


def cross_covariance(x: Union[np.ndarray, List[float]],
                     y: Union[np.ndarray, List[float], None],
                     *,
                     fft: bool = False) -> np.ndarray:
    """Calculate the biased cross covariance estimate between two time series.

    Calculate the cross covariance between two time series for every lag for
    the input arrays. This estimator is biased.

    Args:
        x (array_like, 1d): Time series data.
        y (array_like, 1d): Time series data.
        fft (bool, optional): If True, use FFT convolution. FFT should be
            preferred for long time series. (default: False)

    Returns:
        1darray: The calculated cross covariances.

    """
    if y is None:
        return auto_covariance(x, fft=fft)

    # If x and y are the same object we can save ourselves some computation.
    if y is x:
        return auto_covariance(x, fft=fft)

    x = np.asarray(x)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CRError(msg)

    # Data size
    x_size = x.size

    if x_size < 1:
        msg = 'x is empty.'
        raise CRError(msg)

    if not np.all(np.isfinite(x)):
        msg = 'there is at least one value in the input array x which is '
        msg += 'non-finite or not-number.'
        raise CRError(msg)

    y = np.array(y, copy=False)

    if x.shape != y.shape:
        msg = 'x and y time series should have the same shape.'
        raise CRError(msg)

    if not np.all(np.isfinite(y)):
        msg = 'there is at least one value in the input array y which is '
        msg += 'non-finite or not-number.'
        raise CRError(msg)

    # Fluctuations
    dx = x - x.mean()
    dy = y - y.mean()

    if fft:
        # Find the optimal size for the FFT solver
        optimal_size = get_fft_optimal_size(2 * x_size)

        # Compute the one-dimensional discrete Fourier Transform
        dftx = np.fft.rfft(dx, n=optimal_size)
        dfty = np.fft.rfft(dy, n=optimal_size)
        dftx *= np.conjugate(dfty)

        # Compute the one-dimensional inverse discrete Fourier Transform
        crosscov = np.fft.irfft(dftx, n=optimal_size)[:x_size]

        # Get the real part
        crosscov = crosscov.real
    else:
        # Cross-correlation of two one-dimensional sequences
        crosscov = np.correlate(dx, dy, 'full')[x_size - 1:]

    crosscov /= float(x_size)

    return crosscov


def auto_correlate(x: Union[np.ndarray, List[float]],
                   *,
                   nlags: Optional[int] = None,
                   fft: bool = False) -> np.ndarray:
    """Calculate the auto-correlation function.

    Calculate the auto-correlation function for `nlags` lag for the input
    array. This estimator is biased.

    Args:
        x (array_like, 1d): Time series data.
        nlags (int > 0 or None, optional): Number of lags to return
            auto-correlation for it. (default: None)
        fft (bool, optional): If True, use FFT convolution. FFT should be
            preferred for long time series. (default: False)

    Returns:
        ndarray: The calculated auto correlation function.

    """
    x = np.asarray(x)

    # Calculate (estimate) the auto covariances
    autocor = auto_covariance(x, fft=fft)

    if isclose(autocor[0], 0, abs_tol=_DEFAULT_ABS_TOL):
        msg = 'divide by zero encountered, which means the first element of '
        msg += 'the auto covariances of x is zero (or close to zero).'
        raise CRError(msg)

    if nlags is None:
        # Calculate the auto correlation
        autocor /= autocor[0]
    else:
        # Check input
        if not isinstance(nlags, int):
            msg = 'nlags must be an `int`.'
            raise CRError(msg)

        if nlags < 1:
            msg = 'nlags must be a positive `int`.'
            raise CRError(msg)

        nlags = min(nlags, x.size)

        # Calculate the auto correlation
        autocor = autocor[:nlags + 1] / autocor[0]

    return autocor


def cross_correlate(x: Union[np.ndarray, List[float]],
                    y: Union[np.ndarray, List[float], None],
                    *,
                    nlags: Optional[int] = None,
                    fft: bool = False) -> np.ndarray:
    """Calculate the cross-correlation function.

    Calculate the cross-correlation function for `nlags` lag for the input
    array. This estimator is biased.

    Args:
        x (array_like, 1d): Time series data.
        y (array_like, 1d): Time series data.
        nlags (int > 0 or None, optional): Number of lags to return
            auto-correlation for. (default: None)
        fft (bool, optional): If True, use FFT convolution. FFT should be
            preferred for long time series. (default: False)

    Returns:
        ndarray: The calculated cross correlation.

    """
    if y is None:
        return auto_correlate(x, nlags=nlags, fft=fft)

    # If x and y are the same object we can save ourselves some computation.
    if y is x:
        return auto_correlate(x, nlags=nlags, fft=fft)

    # Calculate the cross covariances
    crosscorr = cross_covariance(x, y, fft=fft)

    x = np.asarray(x)
    y = np.array(y, copy=False)

    sigma_xy = np.std(x) * np.std(y)

    if isclose(sigma_xy, 0, abs_tol=_DEFAULT_ABS_TOL):
        msg = 'Divide by zero encountered, which means the multiplication '
        msg += 'of the standard deviation of x and y is zero.'
        raise CRError(msg)

    if nlags is None:
        # Calculate the cross correlation
        crosscorr /= sigma_xy
    else:
        # Check input
        if not isinstance(nlags, int):
            msg = 'nlags must be an `int`.'
            raise CRError(msg)

        if nlags < 1:
            msg = 'nlags must be a positive `int`.'
            raise CRError(msg)

        nlags = min(nlags, x.size)

        # Calculate the cross correlation
        crosscorr = crosscorr[:nlags + 1] / sigma_xy

    return crosscorr


def modified_periodogram(x: Union[np.ndarray, List[float]],
                         *,
                         fft: bool = False,
                         with_mean: bool = False) -> np.ndarray:
    r"""Compute a modified periodogram to estimate the power spectrum.

    Estimate the power spectrum using a modified periodogram.
    A periodogram [2]_ is an estimate of the spectral density of a signal
    and it is defined as,

    .. math::

        \left \{ I\left(\frac{k}{n}\right) \right \}_{k = 1, \cdots, \left \lfloor \frac{n}{2} \right \rfloor},~ I\left( \frac{k}{n} \right) = \left| \sum_{j=0}^{j=n-1} {x(j) e^{-2\pi i j k / n}} \right|^2 / n

    Args:
        x (array_like, 1d): Time series data.
        fft (bool, optional): If True, use FFT convolution. FFT should be
            preferred for long time series. (default: False)
        with_mean (bool, optional): If True, use x minus its mean.
            (default: False)

    Returns:
        1darray: Computed modified periodogram array.

    Note:
        This function does not return the array of sample frequencies. In
        case of need, one can compute it as,

        .. math::

            f = \left \{ \frac{k}{n} \right \}_{k = 1, \cdots, \left \lfloor \frac{n}{2} \right \rfloor + 1}

        or

        >>> f = np.arange(1., x.size//2 + 1) / x.size

    References:
        .. [2] Heidelberger, P. and Welch, P.D. (1981). "A Spectral Method
               for Confidence Interval Generation and Run Length Control in
               Simulations". Comm. ACM., 24, p. 233--245.

    """
    if with_mean:
        x = np.asarray(x)

        if x.ndim != 1:
            msg = 'x is not an array of one-dimension.'
            raise CRError(msg)

        # Fluctuations
        _mean = np.mean(x)

        if not np.isfinite(_mean):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CRError(msg)

        dx = x - _mean

        del _mean

    else:
        dx = np.array(x, copy=False)

        if dx.ndim != 1:
            msg = 'x is not an array of one-dimension.'
            raise CRError(msg)

        if not np.all(np.isfinite(dx)):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CRError(msg)

    # Data size
    x_size = dx.size

    scale = 1.0 / float(x_size)

    # Perform the fft
    if fft:
        # Compute the one-dimensional discrete Fourier Transform
        result = np.fft.rfft(dx)[1:]
        result *= np.conjugate(result)
    else:
        # The periodogram is defined in [2]_ as,
        # I(k/n) = | \sum_{j=0}^{j=n-1} {x(j) e^{-2\pi i j k / n}} |^2 / n
        # k = 1, n // 2
        arg = np.arange(1, x_size // 2 + 1, dtype=np.float64)
        arg *= scale * 2.0 * pi
        arg = arg * complex(0.0, 1.0)

        k = np.arange(x_size).reshape((1, -1))
        e = arg.reshape((-1, 1)) * k
        e = np.exp(e)

        sumc = e * dx
        sumc = sumc.sum(axis=1)
        result = sumc * np.conjugate(sumc)

    result = result.real
    result *= scale
    return result


def periodogram(x: Union[np.ndarray, List[float]],
                *,
                fft: bool = False,
                with_mean: bool = False) -> np.ndarray:
    r"""Compute a periodogram to estimate the power spectrum.

    Args:
        x (array_like, 1d): Time series data.
        fft (bool, optional): If True, use FFT convolution. FFT should be
            preferred for long time series. (default: False)
        with_mean (bool, optional): If True, use x minus its mean.
            (default: False)

    Returns:
        1darray: Computed power spectrum array.

    Note:
        This function does not return the array of sample frequencies. In
        case of need, one can compute it as,

        .. math::

            f = \left \{ \frac{k}{n} \right \}_{k = 1, \cdots, \left \lfloor \frac{n}{2} \right \rfloor + 1}

        or

        >>> f = np.arange(1., x.size//2 + 1) / x.size

    """
    try:
        result = modified_periodogram(x, fft=fft, with_mean=with_mean)
    except CRError:
        msg = 'Failed to compute a modified periodogram.'
        raise CRError(msg)

    # Data size
    x_size = np.size(x)

    scale = 1.0 / float(x_size)

    result *= scale

    if x_size % 2:
        result *= 2
    else:
        # Do not double the last point since
        # it is an unpaired Nyquist freq point
        result[:-1] *= 2

    return result


def summary(x: Union[np.ndarray, List[float]]):
    """Return the summary of the time series data.

    Args:
        x(array_like, 1d): Time series data.

    Returns:
        float, float, float, float, float, float, float:
            min, max, mean, std, median, 1stQU, 3rdQU

    """
    x = np.asarray(x)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CRError(msg)

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    x_mean = np.nanmean(x)
    x_std = np.nanstd(x)
    x_median = np.nanmedian(x)
    x_1st_quartile, x_3rd_quartile = np.nanquantile(x, [0.25, 0.75])
    return \
        x_min, \
        x_max, \
        x_mean, \
        x_std, \
        x_median, \
        x_1st_quartile, \
        x_3rd_quartile


def int_power(x: Union[np.ndarray, List[float]],
              exponent: int) -> np.ndarray:
    """Array elements raised to the power exponent.

    Args:
        x (array_like, 1d): The bases.
        exponent (int): The exponent

    Returns:
        1darray: Computed power array.

    """
    x = np.asarray(x)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CRError(msg)

    if x.size < 1:
        msg = 'x is empty.'
        raise CRError(msg)

    if not isinstance(exponent, int):
        msg = 'exponent must be an `int`.'
        raise CRError(msg)

    if not np.all(np.isfinite(x)):
        msg = 'there is at least one value in the input array x which is '
        msg += 'non-finite or not-number.'
        raise CRError(msg)

    if exponent == 1:
        return x
    elif exponent == 2:
        return x * x

    nn = exponent if exponent >= 0 else -exponent
    ww = x.copy()
    yy = np.ones(ww.size)
    while nn != 0:
        if nn & 1:
            yy *= ww
        nn >>= 1
        ww *= ww
    if exponent >= 0:
        return yy
    return 1.0 / yy


def moment(x: Union[np.ndarray, List[float]],
           *,
           moment: int = 1) -> float:
    r"""Calculates the nth moment about the mean for a sample.

    Args:
        x (array_like, 1d): Time series data.
        moment (int, optional): Order of central moment that is returned.
            (default: 1)

    Returns:
        float: n-th central moment.

    Note:
        The k-th central moment of a time series data,

        .. math::

            m_k = \frac{1}{n} \sum_{i = 1}^n (x_i - \bar{x})^k,

        where :math:`n` is the number of samples and :math:`\bar{x}` is the
        mean.

    """
    x = np.asarray(x)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CRError(msg)

    if x.size < 1:
        msg = 'x is empty.'
        raise CRError(msg)

    if not np.all(np.isfinite(x)):
        msg = 'there is at least one value in the input array x which is '
        msg += 'non-finite or not-number.'
        raise CRError(msg)

    if not isinstance(moment, int):
        msg = 'moment must be an `int`.'
        raise CRError(msg)

    # The first moment about the mean is 0
    if moment == 1:
        return 0.0

    dx = x - x.mean()
    dx_power_moment = int_power(dx, moment)
    return dx_power_moment.mean()


def skew(x: Union[np.ndarray, List[float]],
         *,
         bias: bool = False) -> float:
    r"""Compute the time series data set skewness.

    ``skewness`` is a measure of the asymmetry of the probability distribution
    of a real-valued random variable about its mean.

    Args:
        x (array_like, 1d): Time series data.
        bias (bool, optional): If False, then the calculations are corrected
            for statistical bias. (default: False)

    Returns:
        float: The skewness

    Note:
        For normally distributed data, the skewness should be about zero.
        For unimodal continuous distributions, a skewness value greater than
        zero means that there is more weight in the right tail of the
        distribution.

        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness :math:`g_1 = \frac{m_3}{m_2^{3/2}}`, where :math:`m_i` is
        the biased sample :math:`i\texttt{th}` central moment. If ``bias`` is
        False, the calculations are corrected for bias and the value computed
        is the adjusted Fisher-Pearson standardized moment coefficient, i.e.

        .. math::

            G_1 = \frac{k_3}{k_2^{3/2}} = \frac{\sqrt{N(N-1)}}{N-2} \frac{m_3}{m_2^{3/2}}.


    References:
        .. [13] Zwillinger, D. and Kokoska, S., (2000). "CRC Standard
                Probability and Statistics Tables and Formulae," Chapman &
                Hall:New York. 2000. Section 2.2.24.1

    """
    x_size = np.size(x)
    if x_size in (1, 2):
        return 0.0
    moment_2 = moment(x, moment=2)
    if moment_2 != 0:
        moment_3 = moment(x, moment=3)
        val = moment_3 / moment_2 ** 1.5
        if bias:
            return val
    else:
        return 0.0
    fac = sqrt((x_size - 1.0) * x_size) / (x_size - 2.0)
    return fac * val
