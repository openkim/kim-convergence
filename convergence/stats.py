"""Stats module."""

from math import isclose
import numpy as np
from bisect import bisect_left

from .err import CVGError

__all__ = [
    'get_fft_optimal_size',
    'auto_covariance',
    'auto_correlate',
    'cross_covariance',
    'cross_correlate',
    'translate_scale',
    'standard_scale',
    'robust_scale',
    'periodogram',
    'outlier_methods',
    'outlier_test'
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


def get_fft_optimal_size(input_size: int):
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
        raise CVGError(msg)

    if input_size < 7:
        if input_size < 0:
            msg = 'input_size must be a positive `int`.'
            raise CVGError(msg)
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


def auto_covariance(x, *, fft=False):
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
        :math:`\frca{1}{N}` formulation has some desirable statistical
        properties and is the most commonly used in the statistics literature.

    Args:
        x (array_like, 1d): Time series data.
        fft (bool, optional): If True, use FFT convolution. FFT should be
            preferred for long time series. (default: False)

    Returns:
        1darray: The estimated autocovariances.

    """
    x = np.array(x, copy=False)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    # Data size
    x_size = x.size

    if x_size < 1:
        msg = 'x is empty.'
        raise CVGError(msg)

    if not np.all(np.isfinite(x)):
        msg = 'there is at least one value in the input array which is '
        msg += 'non-finite or not-number.'
        raise CVGError(msg)

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


def cross_covariance(x, y, *, fft=False):
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

    x = np.array(x, copy=False)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    # Data size
    x_size = x.size

    if x_size < 1:
        msg = 'x is empty.'
        raise CVGError(msg)

    if not np.all(np.isfinite(x)):
        msg = 'there is at least one value in the input array x which is '
        msg += 'non-finite or not-number.'
        raise CVGError(msg)

    y = np.array(y, copy=False)

    if x.shape != y.shape:
        msg = 'x and y time series should have the same shape.'
        raise CVGError(msg)

    if not np.all(np.isfinite(y)):
        msg = 'there is at least one value in the input array y which is '
        msg += 'non-finite or not-number.'
        raise CVGError(msg)

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


def auto_correlate(x, *, nlags=None, fft=False):
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
    x = np.array(x, copy=False)

    # Calculate (estimate) the auto covariances
    autocor = auto_covariance(x, fft=fft)

    # assures that the two values are the same within about 14 decimal digits.
    if isclose(autocor[0], 0, rel_tol=1e-14):
        msg = 'divide by zero encountered, which means the first element of '
        msg += 'the auto covariances of x is zero (or close to zero).'
        raise CVGError(msg)

    if nlags is None:
        # Calculate the auto correlation
        autocor /= autocor[0]
    else:
        # Check input
        if not isinstance(nlags, int):
            msg = 'nlags must be an `int`.'
            raise CVGError(msg)

        if nlags < 1:
            msg = 'nlags must be a positive `int`.'
            raise CVGError(msg)

        nlags = min(nlags, x.size)

        # Calculate the auto correlation
        autocor = autocor[:nlags + 1] / autocor[0]

    return autocor


def cross_correlate(x, y, *, nlags=None, fft=False):
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

    x = np.array(x, copy=False)
    y = np.array(y, copy=False)

    sigma_xy = np.std(x) * np.std(y)

    # assures that the two values are the same within about 14 decimal digits.
    if isclose(sigma_xy, 0, rel_tol=1e-14):
        msg = 'Divide by zero encountered, which means the multiplication '
        msg += 'of the standard deviation of x and y is zero.'
        raise CVGError(msg)

    if nlags is None:
        # Calculate the cross correlation
        crosscorr /= sigma_xy
    else:
        # Check input
        if not isinstance(nlags, int):
            msg = 'nlags must be an `int`.'
            raise CVGError(msg)

        if nlags < 1:
            msg = 'nlags must be a positive `int`.'
            raise CVGError(msg)

        nlags = min(nlags, x.size)

        # Calculate the cross correlation
        crosscorr = crosscorr[:nlags + 1] / sigma_xy

    return crosscorr


def translate_scale(x, *, with_centering=True, with_scaling=True):
    r"""Standardize a dataset.

    Standardize a dataset by translating the data set so that :math:`x[0]=0`
    and rescaled by overall averages so that the numbers are of O(1) with a
    good spread. (default: True)

    The translate and scale of a sample `x` is calculated as:

    .. math::

        z = \frac{(x - x_0)}{u}

    where :math:`x_0` is :math:`x[0]` or :math:`0` if `with_centering=False`,
    and `u` is the mean of the samples or :math:`1` if `with_scaling=False`.

    Args:
        x (array_like, 1d): The data to center and scale.
        with_centering (bool, optional): If True, use x minus its first
            element. (default: True)
        with_scaling (bool, optional): If True, scale the data to overall
            averages so that the numbers are of O(1) with a good spread.
            (default: True)

    Returns:
        1darray: Scaled dataset

    """
    x = np.array(x, copy=False)

    if with_centering:
        # Fluctuations
        dx = x - x[0]
    else:
        dx = np.array(x, copy=True)

    if with_scaling:
        mean_ = np.mean(dx)

        if not np.isfinite(mean_):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CVGError(msg)

        # assures that the two values are not the same
        # within about 14 decimal digits.
        if not isclose(mean_, 0, rel_tol=1e-14):
            dx /= mean_

    return dx


def standard_scale(x, *, with_centering=True, with_scaling=True):
    r"""Standardize a dataset.

    Standardize a dataset by removing the mean and scaling to unit variance.
    The standard score of a sample `x` is calculated as:

    .. math::

        z = \frac{(x - u)}{s}

    where `u` is the mean of the samples or :math:`0` if `with_centering=False`
    , and `s` is the standard deviation of the samples or :math:`1` if
    `with_scaling=False`.

    Args:
        x (array_like, 1d): The data to center and scale.
        with_centering (bool, optional): If True, use x minus its mean, or
            center the data before scaling. (default: True)
        with_scaling (bool, optional): If True, scale the data to unit
            variance (or equivalently, unit standard deviation).
            (default: True)

    Returns:
        1darray: scaled dataset

    Notes:
        If set explicitly `with_centering=False` (only variance scaling will
        be performed on x). We use a biased estimator for the standard
        deviation.

    """
    x = np.array(x, copy=False)

    if with_centering:
        # Fluctuations
        dx = x - np.mean(x)

        mean_1 = np.mean(dx)

        # Verify that mean_1 is 'close to zero'. If x contains very
        # large values, mean_1 can also be very large, due to a lack of
        # dx is a view on the original array
        # Numerical issues were encountered when centering the data
        # and might not be solved. Dataset may contain too large values.
        # You may need to prescale your features.

        if not np.isfinite(mean_1):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CVGError(msg)

        # assures that the two values are not the same
        # within about 14 decimal digits.
        if not isclose(mean_1, 0, rel_tol=1e-14):
            dx -= mean_1

    else:
        dx = np.array(x, copy=True)

    if with_scaling:
        scale_ = np.std(x)

        if not np.isfinite(scale_):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CVGError(msg)

        # assures that the two values are not the same
        # within about 14 decimal digits.
        if not isclose(scale_, 0, rel_tol=1e-14):
            dx /= scale_

        if with_centering:
            mean_2 = np.mean(dx)

            # If mean_2 is not 'close to zero', it comes from the fact that
            # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
            # if mean_1 was close to zero. The problem is thus essentially
            # due to the lack of precision of np.mean(x). A solution is then
            # to subtract the mean again.
            # Numerical issues were encountered when centering the data
            # and might not be solved. Dataset may contain too large values.
            # You may need to prescale your features.

            # assures that the two values are not the same
            # within about 14 decimal digits.
            if not isclose(mean_2, 0, rel_tol=1e-14):
                dx -= mean_2

    return dx


def robust_scale(x,
                 *,
                 with_centering=True,
                 with_scaling=True,
                 quantile_range=(25.0, 75.0)):
    """Standardize a dataset.

    Standardize a dataset by centering to the median and component wise scale
    according to the inter-quartile range.

    Args:
        x (array_like, 1d): The data to center and scale.
        with_centering (bool, optional): If True, center the data before
            scaling. (default: True)
        with_scaling (bool, optional): If True, scale the data.
            (default: True)
        quantile_range (tuple, or list, optional): (q_min, q_max),
            0.0 < q_min < q_max < 100.0
            (default: (25.0, 75.0) = (1st quantile, 3rd quantile))

    Returns:
        1darray: scaled dataset

    """
    x = np.array(x, copy=False)

    if not isinstance(quantile_range, tuple) or \
            not isinstance(quantile_range, list):
        msg = 'invalid quantile range: {}.'.format(str(quantile_range))
        raise CVGError(msg)

    if len(quantile_range) != 2:
        msg = 'invalid quantile range: {}.'.format(str(quantile_range))
        raise CVGError(msg)

    q_min, q_max = quantile_range
    if not 0 <= q_min <= q_max <= 100:
        msg = 'invalid quantile range: {}.'.format(str(quantile_range))
        raise CVGError(msg)

    if with_centering:
        dx = x - np.median(x)
    else:
        dx = np.array(x, copy=True)

    if with_scaling:
        quantiles = np.percentile(x, quantile_range)

        scale_ = quantiles[1] - quantiles[0]

        if not np.isfinite(scale_):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CVGError(msg)

        # assures that the two values are not the same
        # within about 14 decimal digits.
        if not isclose(scale_, 0, rel_tol=1e-14):
            dx /= scale_

    return dx


def periodogram(x, *, fft=False, with_mean=False):
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
    x = np.array(x, copy=False)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    if with_mean:
        # Fluctuations
        _mean = np.mean(x)

        if not np.isfinite(_mean):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CVGError(msg)

        dx = x - _mean

        del _mean

    else:
        if not np.all(np.isfinite(x)):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CVGError(msg)

        dx = np.array(x, copy=False)

    # Data size
    x_size = x.size

    scale = 1.0 / float(x_size)
    scale2 = scale * scale

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
        arg *= scale * 2.0 * np.pi
        arg = arg * complex(0.0, 1.0)

        k = np.arange(x_size).reshape([1, -1])
        e = arg.reshape([-1, 1]) * k
        e = np.exp(e)

        sumc = e * x
        sumc = sumc.sum(axis=1)
        result = sumc * np.conjugate(sumc)

    result = result.real

    result *= scale2

    if x_size % 2:
        result *= 2
    else:
        # Do not double the last point since
        # it is an unpaired Nyquist freq point
        result[:-1] *= 2

    return result


def summary(x):
    """Return the summary of the time series data.

    Args:
        x (array_like, 1d): Time series data.

    Returns:
        float, float, float, float, float, float, float :
            min, max, mean, std, median, 1stQU, 3rdQU

    """
    x = np.array(x, copy=False)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

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


outlier_methods = ('iqr', 'boxplot',
                   'extreme_iqr', 'extreme_boxplot',
                   'z_score', 'standard_score',
                   'modified_z_score')
"""Methods to decide what are outliers in the data.

- iqr or boxplot:
  The interquartile range ('iqr') or 'boxplot' is a graphical display for
  illustrating the data's behavior in the middle and at the distributions'
  ends. The lower and upper quartiles Q1 and Q3, respectively (defined as the
  25th and 75th percentiles) and the difference (Q3 - Q1) is called the
  interquartile range.

  In the 'iqr' or 'boxplot' method, a point beyond an inner-fence on either
  side is considered an outlier. The inner-fence is the multiplication of '1.5'
  and the interquartile range (:math:`1.5 \times \text{IQR}`).

- extreme_iqr or extreme_boxplot:
  In the `extreme_iqr` method, a point beyond an outer-fence is considered an
  outlier. The outer-fence is the multiplication of 3 and the interquartile
  range (:math:`3 \times \text{IQR}`).

- z_score or standard_score:
    The 'z_score', or 'standard_score', describes a point compared to the
    data's mean and standard deviation. The goal is to remove the effects of
    the location and scale of the data. This approach maps the data onto a
    distribution whose mean is 0 and whose standard deviation is 1. Anything
    that is too far from zero (the threshold is 3 or -3) is considered an
    outlier.

- modified_z_score:
    'modified_z_score' is a method to improve the 'z_score' approach in small
    datasets (usually when the dataset has fewer than 12 points). It uses the
    median and MAD rather than the mean and standard deviation, which are
    robust central tendency and dispersion measures, respectively.

"""


def outlier_test(x, outlier_method='iqr'):
    """Test to detect what are outliers in the data.

    The intuitive definition for the concept of an outlier in the data is a
    point that significantly deviates from its expected value. Therefore, given
    a time series (or a random sample from a population), a point can be
    declared an outlier if the distance to its expected value is higher than a
    predefined threshold (:math:`|x_i - E(x)| > \tau`), where :math:`x_i` is
    the observed data point, and :math:`E(x)` is its expected value.

    The methods based on this strategy are the most common approaches in the
    literature. These methods intend to detect outliers, but it is up to the
    analyst to decide if the detected points are real outliers. Thus it is
    necessary to characterize standard data points before removing any outliers
    detected by these approaches.

    Args:
        x (array_like, 1d): Time series data.
        method (str, optional): Method to detect what are outliers in the data.
            (default: 'iqr')

    Returns:
        1darray, or None: outliers indices, or `None` if there is no outlier

    """
    x = np.array(x, copy=False)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    if not np.all(np.isfinite(x)):
        msg = 'there is at least one value in the input array which is '
        msg += 'non-finite or not-number.'
        raise CVGError(msg)

    if isinstance(outlier_method, str):
        if outlier_method not in outlier_methods:
            msg = 'method {} not found. Valid '.format(outlier_method)
            msg += 'methods to detect outliers are:\n\t- '
            msg += '{}'.format('\n\t- '.join(outlier_methods))
            raise CVGError(msg)
    else:
        msg = 'Input outlier_method is not a `str`.'
        raise CVGError(msg)

    if outlier_method in ('iqr', 'boxplot'):
        lower_quartile, upper_quartile = np.quantile(x, [0.25, 0.75])
        difference = upper_quartile - lower_quartile
        lower_inner_fence = lower_quartile - (difference * 1.5)
        upper_inner_fence = upper_quartile + (difference * 1.5)
        outliers_indices = np.where(
            (x < lower_inner_fence) | (x > upper_inner_fence))
    elif outlier_method in ('extreme_iqr', 'extreme_boxplot'):
        lower_quartile, upper_quartile = np.quantile(x, [0.25, 0.75])
        difference = upper_quartile - lower_quartile
        lower_outer_fence = lower_quartile - (difference * 3.0)
        upper_outer_fence = upper_quartile + (difference * 3.0)
        outliers_indices = np.where(
            (x < lower_outer_fence) | (x > upper_outer_fence))
    elif outlier_method in ('z_score', 'standard_score'):
        x_mean = np.mean(x)
        x_std = np.std(x)
        z_score = (x - x_mean) / x_std
        outliers_indices = np.where(np.abs(z_score) > 3)
    elif outlier_method == 'modified_z_score':
        x_median = np.median(x)
        x_median_absolute_deviation = np.median(np.abs(x - x_median))
        modified_z_score = 0.6745 * \
            (x - x_median) / x_median_absolute_deviation
        outliers_indices = np.where(np.abs(modified_z_score) > 3.5)

    if np.size(outliers_indices):
        return outliers_indices[0]

    return None
