"""Upper Confidence Limit (UCL) module."""

import numpy as np
from numpy.linalg import pinv, norm, inv

from .err import CVGError
from .batch import batch
from .stats import periodogram
from .t_dist import t_inv_cdf
from .utils import train_test_split

__all__ = [
    'set_heidel_welch_constants',
    'get_heidel_welch_constants',
    'get_heidel_welch_set',
    'get_heidel_welch_knp',
    'get_heidel_welch_A',
    'get_heidel_welch_C1',
    'get_heidel_welch_C2',
    'get_heidel_welch_tm',
    'ucl',
]


# Heidelberger and Welch (1981)
#
# Heidelberger, P. and Welch, P.D. (1981). "A Spectral Method for Confidence
# Interval Generation and Run Length Control in Simulations". Comm. ACM., 24,
# p. 233--245.
heidel_welch_set = False
"""bool: Flag indicating if the Heidelberger and Welch constants are set."""

heidel_welch_k = None
"""int: The number of points that are used to obtain the polynomial fit in Heidelberger and Welch's spectral method."""
heidel_welch_n = None
"""int: The number of time series data points or number of batches in Heidelberger and Welch's spectral method."""
heidel_welch_p = None
"""float: Probability."""

A = None
"""array_2d: Auxiliary matrix."""

Aplus_1 = None
"""array_2d: The (Moore-Penrose) pseudo-inverse of a matrix for the first degree polynomial fit in Heidelberger and Welch's spectral method."""
Aplus_2 = None
"""array_2d: The (Moore-Penrose) pseudo-inverse of a matrix for the second degree polynomial fit in Heidelberger and Welch's spectral method."""
Aplus_3 = None
"""array_2d: The (Moore-Penrose) pseudo-inverse of a matrix for the third degree polynomial fit in Heidelberger and Welch's spectral method."""

heidel_welch_C1_1 = None
"""float: Heidelberger and Welch's C1 constant for the first degree polynomial fit."""
heidel_welch_C1_2 = None
"""float: Heidelberger and Welch's C1 constant for the second degree polynomial fit."""
heidel_welch_C1_3 = None
"""float: Heidelberger and Welch's C1 constant for the third degree polynomial fit."""

heidel_welch_C2_1 = None
"""int: Heidelberger and Welch's C2 constant for the first degree polynomial fit."""
heidel_welch_C2_2 = None
"""int: Heidelberger and Welch's C2 constant for the first degree polynomial fit."""
heidel_welch_C2_3 = None
"""int: Heidelberger and Welch's C2 constant for the first degree polynomial fit."""

tm_1 = None
"""float: t_distribution inverse cumulative distribution function for C2_1 degrees of freedom."""
tm_2 = None
"""float: t_distribution inverse cumulative distribution function for C2_2 degrees of freedom."""
tm_3 = None
"""float: t_distribution inverse cumulative distribution function for C2_3 degrees of freedom."""


def set_heidel_welch_constants(p=0.975, k=50):
    """Set Heidelberger and Welch constants globally.

    Set the constants necessary for application of the Heidelberger and
    Welch's [2]_ confidence interval generation method.

    Keyword Args:
        p (float, optional): probability (or confidence interval) and must be
            between 0.0 and 1.0. (default: 0.975)
        k (int, optional): the number of points in Heidelberger and Welch's
            spectral method that are used to obtain the polynomial fit. The
            parameter ``k`` determines the frequency range over which the fit
            is made. (default: 50)

    """
    global heidel_welch_set
    global heidel_welch_k
    global heidel_welch_n
    global heidel_welch_p
    global A
    global Aplus_1
    global Aplus_2
    global Aplus_3
    global heidel_welch_C1_1
    global heidel_welch_C1_2
    global heidel_welch_C1_3
    global heidel_welch_C2_1
    global heidel_welch_C2_2
    global heidel_welch_C2_3
    global tm_1, tm_2, tm_3

    if p <= 0.0 or p >= 1.0:
        msg = 'probability (or confidence interval) p = {} '.format(p)
        msg += 'is not in the range (0.0 1.0).'
        raise CVGError(msg)

    if heidel_welch_set and k == heidel_welch_k:
        if p != heidel_welch_p:
            tm_1 = t_inv_cdf(p, heidel_welch_C2_1)
            tm_2 = t_inv_cdf(p, heidel_welch_C2_2)
            tm_3 = t_inv_cdf(p, heidel_welch_C2_3)
            heidel_welch_p = p
        return

    if isinstance(k, int):
        if k < 25:
            msg = 'wrong number of points k = {} is given to '.format(k)
            msg = 'obtain the polynomial fit. According to Heidelberger, '
            msg += 'and Welch, (1981), this procedure at least needs to '
            msg += 'have 25 points.'
            raise CVGError(msg)
    else:
        msg = 'k is the number of points and should be a positive `int`.'
        raise CVGError(msg)

    heidel_welch_k = k
    heidel_welch_n = k * 4
    heidel_welch_p = p

    # Auxiliary matrix
    f = np.arange(1, heidel_welch_k + 1) * 4 - 1.0
    f /= (2.0 * heidel_welch_n)

    A = np.empty((heidel_welch_k, 4), dtype=np.float64)

    A[:, 0] = np.ones((heidel_welch_k), dtype=np.float64)
    A[:, 1] = f
    A[:, 2] = f * f
    A[:, 3] = A[:, 2] * f

    # The (Moore-Penrose) pseudo-inverse of a matrix.
    # Calculate the generalized inverse of a matrix using its singular-value
    # decomposition (SVD) and including all large singular values.
    Aplus_1 = pinv(A[:, :2])
    Aplus_2 = pinv(A[:, :3])
    Aplus_3 = pinv(A)

    # Heidelberger and Welch (1981) constants Table 1
    _sigma2 = 0.645 * inv(np.dot(np.transpose(A[:, :2]), A[:, :2]))[0, 0]
    heidel_welch_C1_1 = np.exp(-_sigma2 / 2.)
    # Heidelberger and Welch's C2 constant for the first degree polynomial fit.
    heidel_welch_C2_1 = int(np.rint(2. / (np.exp(_sigma2) - 1.)))

    _sigma2 = 0.645 * inv(np.dot(np.transpose(A[:, :3]), A[:, :3]))[0, 0]
    heidel_welch_C1_2 = np.exp(-_sigma2 / 2.)
    # Heidelberger and Welch's C2 constant for the second degree polynomial fit.
    heidel_welch_C2_2 = int(np.rint(2. / (np.exp(_sigma2) - 1.)))

    _sigma2 = 0.645 * inv(np.dot(np.transpose(A), A))[0, 0]
    heidel_welch_C1_3 = np.exp(-_sigma2 / 2.)
    # Heidelberger and Welch's C2 constant for the third degree polynomial fit.
    heidel_welch_C2_3 = int(np.rint(2. / (np.exp(_sigma2) - 1.)))

    tm_1 = t_inv_cdf(p, heidel_welch_C2_1)
    tm_2 = t_inv_cdf(p, heidel_welch_C2_2)
    tm_3 = t_inv_cdf(p, heidel_welch_C2_3)

    # Set the flag
    heidel_welch_set = True


def get_heidel_welch_constants():
    """Get the Heidelberger and Welch constants."""
    return \
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
        tm_3


def get_heidel_welch_set():
    """Get the Heidelberger and Welch setting flag."""
    return heidel_welch_set


def get_heidel_welch_knp():
    """Get the Heidelberger and Welch k, n, and p constants."""
    return heidel_welch_k, heidel_welch_n, heidel_welch_p


def get_heidel_welch_A():
    """Get the Heidelberger and Welch auxilary matrices."""
    return A, Aplus_1, Aplus_2, Aplus_3


def get_heidel_welch_C1():
    """Get the Heidelberger and Welch C1 constants."""
    return heidel_welch_C1_1, heidel_welch_C1_2, heidel_welch_C1_3


def get_heidel_welch_C2():
    """Get the Heidelberger and Welch C2 constants."""
    return heidel_welch_C2_1, heidel_welch_C2_2, heidel_welch_C2_3


def get_heidel_welch_tm():
    """Get the Heidelberger and Welch t_distribution ppf for C2 degrees of freedom."""
    return tm_1, tm_1, tm_1


def ucl(x, *, p=0.975, k=50, fft=True, test_size=None, train_size=None):
    r"""Approximate the upper confidence limit of the mean.

    Approximate an unbiased estimate of the upper confidence limit or
    half the width of the p% probability interval (confidence interval, or
    credible interval) around the time-series mean.

    An estimate of the variance of the time-series mean is obtained by
    estimating the spectral density at zero frequency [12]_. We use an
    adaptive method which select the degree of the polynomial according to
    the shape of the periodogram [2]_.

    The estimated halfwidth of the confidence interval of time-series mean
    is computed as :math:`\frac{UCL}{\hat{\mu}}.`
    Where, UCL is the upper confidence limit, and :math:`\hat{\mu}` is the
    time-series mean.

    The upper confidence limit can be computed as,

    .. math::

        UCL = t_m\left(\text{p}\right)\left(\hat{P}(0)/N\right)^{1/2},

    where :math:`N` is the number of data points, and :math:`t` is a
    t-distribution with :math:`m=C_2` degrees of freedom.

    For :math:`\text{p} = 0.95`, or with the chance about 95% (and 10
    degrees of freedom) the t-value is 1.812. It implies that the true mean
    is lying between :math:`\hat{\mu} \pm UCL` with chance about 95%.
    In other words, the probability that the true mean lies between the upper
    and lower threshold is 95%.

    Args:
        x {array_like, 1d}: time series data.

    Keyword Args:
        p (float, optional): probability (or confidence interval) and must be
            between 0.0 and 1.0, and represents the confidence for calculation
            of relative halfwidths estimation. (default: 0.975)
        k (int, optional): the number of points that are used to obtain the
            polynomial fit. The parameter ``k`` determines the frequency range
            over which the fit is made. (default: 50)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        test_size (int, float, optional): if ``float``, should be between 0.0
          and 1.0 and represent the proportion of the periodogram dataset to
          include in the test split. If ``int``, represents the absolute number
          of test samples. (default: None)
        train_size (int, float, optional): if ``float``, should be between 0.0
          and 1.0 and represent the proportion of the preiodogram dataset to
          include in the train split. If ``int``, represents the absolute
          number of train samples. (default: None)

    Returns:
        float: upper_confidence_limit
            The approximately unbiased estimate of variance of the sample mean,
            based on the degree of the fitted polynomial.

    Note:
        if both test_size and train_size are ``None`` it does not split the
        priodogram data to train and test.

    References:
        .. [12] Heidelberger, P. and Welch, P.D. (1983). "Simulation Run
               Length Control in the Presence of an Initial Transient".
               Operations Research, 31(6), p. 1109--1144.

    """
    x = np.array(x, copy=False)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    # We compute once and use it during iterations
    if not heidel_welch_set or k != heidel_welch_k or p != heidel_welch_p:
        set_heidel_welch_constants(p=p, k=k)

    batch_size = x.size // heidel_welch_n

    if batch_size < 1:
        msg = 'not enough data points (batching of the data is not '
        msg += 'possible).\nThe input time series has '
        msg += '{} data points which is smaller than the '.format(x.size)
        msg += 'minimum number of required points = '
        msg += '{} for batching.'.format(heidel_welch_n)
        raise CVGError(msg)

    # Batch the data
    z = batch(x,
              batch_size=batch_size,
              with_centering=False,
              with_scaling=False)

    n_batches = z.size

    if n_batches != heidel_welch_n:
        if n_batches > heidel_welch_n:
            n_batches = heidel_welch_n
            z = z[:n_batches]
        else:
            msg = 'batching of the time series failed. (or there is not '
            msg += 'enough data points)\n'
            msg += 'Number of batches = {} '.format(n_batches)
            msg += 'must be the same as {}.'.format(heidel_welch_n)
            raise CVGError(msg)

    # Compute the periodogram of the sequence z
    period = periodogram(z, fft=fft, with_mean=False)

    l = range(0, period.size, 2)
    r = range(1, period.size, 2)

    # Compute the log of the average of adjacent pefiodogram values
    g = period[l] + period[r]
    g *= 0.5
    g = np.log(g)
    g += 0.27

    # Using ordinary least squares, and fit a polynomial to the data

    if test_size is None and train_size is None:
        # Least-squares solution
        x1 = np.matmul(Aplus_1, g)
        # Error of solution ||g - A*x||
        eps1 = norm(g - np.matmul(A[:, :2], x1))

        # Least-squares solution
        x2 = np.matmul(Aplus_2, g)
        # Error of solution ||g - A*x||
        eps2 = norm(g - np.matmul(A[:, :3], x2))

        # Least-squares solution
        x3 = np.matmul(Aplus_3, g)
        # Error of solution ||g - A*x||
        eps3 = norm(g - np.matmul(A, x3))
    else:
        ind_train, ind_test = train_test_split(
            g, test_size=test_size, train_size=train_size)

        # Least-squares solution
        x1 = np.matmul(Aplus_1[:, ind_train], g[ind_train])
        # Error of solution ||g - A*x||
        eps1 = norm(g[ind_test] - np.matmul(A[ind_test, :2], x1))

        # Least-squares solution
        x2 = np.matmul(Aplus_2[:, ind_train], g[ind_train])
        # Error of solution ||g - A*x||
        eps2 = norm(g[ind_test] - np.matmul(A[ind_test, :3], x2))

        # Least-squares solution
        x3 = np.matmul(Aplus_3[:, ind_train], g[ind_train])
        # Error of solution ||g - A*x||
        eps3 = norm(g[ind_test] - np.matmul(A[ind_test, :], x3))

    # Find the best fit
    d = np.argmin((eps1, eps2, eps3))

    if d == 0:
        # get x, which is an unbiased estimate of log(p(0)).
        x = x1[0]
        C1 = heidel_welch_C1_1
        tm = tm_1
    elif d == 1:
        # get x, which is an unbiased estimate of log(p(0)).
        x = x2[0]
        C1 = heidel_welch_C1_2
        tm = tm_2
    else:
        # get x, which is an unbiased estimate of log(p(0)).
        x = x3[0]
        C1 = heidel_welch_C1_3
        tm = tm_3

    # The variance of the sample mean of a covariance stationary sequence is
    # given approximately by p(O)/N, the spectral density at zero frequency
    # divided by the sample size.

    # Calculate the approximately unbiased estimate of variance of the sample
    # mean
    sigma_sq = C1 * np.exp(x) / float(n_batches)

    upper_confidence_limit = tm * np.sqrt(sigma_sq)
    return upper_confidence_limit
