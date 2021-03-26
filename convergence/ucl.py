"""Upper Confidence Limit (UCL) module.

Upper Confidence Limit (UCL): The upper boundary (or limit) of a confidence
interval of a parameter of interest such as the population mean.
"""

import numpy as np
from numpy.linalg import pinv, norm, inv

from .err import CVGError, cvg_warning
from .batch import batch
from .stats import modified_periodogram
from .t_dist import t_inv_cdf
from .utils import train_test_split, subsample_index

__all__ = [
    'HeidelbergerWelch',
    'ucl',
    'subsamples_ucl',
    'ucl_methods',
]


class HeidelbergerWelch:
    """Heidelberger and Welch class.

    Heidelberger and Welch (1981) [2]_ Object.

    Attributes:
        heidel_welch_set (bool): Flag indicating if the Heidelberger and Welch
            constants are set.
        heidel_welch_k (int) : The number of points that are used to obtain the
            polynomial fit in Heidelberger and Welch's spectral method.
        heidel_welch_n (int) : The number of time series data points or number
            of batches in Heidelberger and Welch's spectral method.
        heidel_welch_p (float) : Probability.
        a_matrix (ndarray) : Auxiliary matrix.
        a_matrix_1_inv (ndarray) : The (Moore-Penrose) pseudo-inverse of a
            matrix for the first degree polynomial fit in Heidelberger and
            Welch's spectral method.
        a_matrix_2_inv (ndarray) : The (Moore-Penrose) pseudo-inverse of a
            matrix for the second degree polynomial fit in Heidelberger and
            Welch's spectral method.
        a_matrix_3_inv (ndarray) : The (Moore-Penrose) pseudo-inverse of a
            matrix for the third degree polynomial fit in Heidelberger and
            Welch's spectral method.
        heidel_welch_c1_1 (float) : Heidelberger and Welch's C1 constant for
            the first degree polynomial fit.
        heidel_welch_c1_2 (float) : Heidelberger and Welch's C1 constant for
            the second degree polynomial fit.
        heidel_welch_c1_3 (float) : Heidelberger and Welch's C1 constant for
            the third degree polynomial fit.
        heidel_welch_c2_1 (float) : Heidelberger and Welch's C2 constant for
            the first degree polynomial fit.
        heidel_welch_c2_2 (float) : Heidelberger and Welch's C2 constant for
            the second degree polynomial fit.
        heidel_welch_c2_3 (float) : Heidelberger and Welch's C2 constant for
            the third degree polynomial fit.
        tm_1 (float) : t_distribution inverse cumulative distribution function
            for C2_1 degrees of freedom.
        tm_2 (float) : t_distribution inverse cumulative distribution function
            for C2_2 degrees of freedom.
        tm_3 (float) : t_distribution inverse cumulative distribution function
            for C2_3 degrees of freedom.

    """

    def __init__(self, *,
                 confidence_coefficient=0.95,
                 heidel_welch_number_points=50):
        """Initialize the class.

        Initialize a HeidelbergerWelch object and set the constants.

        Keyword Args:
            confidence_coefficient (float, optional): probability (or
                confidence interval) and must be between 0.0 and 1.0.
                (default: 0.95)
            heidel_welch_number_points (int, optional): the number of points in
                Heidelberger and Welch's spectral method that are used to
                obtain the polynomial fit. The parameter
                ``heidel_welch_number_points`` determines the frequency range
                over which the fit is made. (default: 50)

        """
        self.heidel_welch_set = False
        self.heidel_welch_k = None
        self.heidel_welch_n = None
        self.heidel_welch_p = None
        self.a_matrix = None
        self.a_matrix_1_inv = None
        self.a_matrix_2_inv = None
        self.a_matrix_3_inv = None
        self.heidel_welch_c1_1 = None
        self.heidel_welch_c1_2 = None
        self.heidel_welch_c1_3 = None
        self.heidel_welch_c2_1 = None
        self.heidel_welch_c2_2 = None
        self.heidel_welch_c2_3 = None
        self.tm_1 = None
        self.tm_2 = None
        self.tm_3 = None
        try:
            self.set_heidel_welch_constants(
                confidence_coefficient=confidence_coefficient,
                heidel_welch_number_points=heidel_welch_number_points)
        except CVGError:
            msg = "Failed to set the Heidelberger and Welch constants."
            raise CVGError(msg)

    def set_heidel_welch_constants(self, *,
                                   confidence_coefficient=0.95,
                                   heidel_welch_number_points=50):
        """Set Heidelberger and Welch constants globally.

        Set the constants necessary for application of the Heidelberger and
        Welch's [2]_ confidence interval generation method.

        Keyword Args:
            confidence_coefficient (float, optional): probability (or
                confidence interval) and must be between 0.0 and 1.0.
                (default: 0.95)
            heidel_welch_number_points (int, optional): the number of points in
                Heidelberger and Welch's spectral method that are used to
                obtain the polynomial fit. The parameter
                ``heidel_welch_number_points`` determines the
                frequency range over which the fit is made. (default: 50)

        """
        if confidence_coefficient <= 0.0 or confidence_coefficient >= 1.0:
            msg = 'probability (or confidence interval) '
            msg += 'confidence_coefficient = '
            msg += '{} '.format(confidence_coefficient)
            msg += 'is not in the range (0.0 1.0).'
            raise CVGError(msg)

        if self.heidel_welch_set and \
                heidel_welch_number_points == self.heidel_welch_k:
            if confidence_coefficient != self.heidel_welch_p:
                self.tm_1 = t_inv_cdf(
                    confidence_coefficient, self.heidel_welch_c2_1)
                self.tm_2 = t_inv_cdf(
                    confidence_coefficient, self.heidel_welch_c2_2)
                self.tm_3 = t_inv_cdf(
                    confidence_coefficient, self.heidel_welch_c2_3)
                self.heidel_welch_p = confidence_coefficient
            return

        if isinstance(heidel_welch_number_points, int):
            if heidel_welch_number_points < 25:
                msg = 'wrong number of points heidel_welch_number_points = '
                msg += '{} is '.format(heidel_welch_number_points)
                msg = 'given to obtain the polynomial fit. According to '
                msg += 'Heidelberger, and Welch, (1981), this procedure '
                msg += 'at least needs to have 25 points.'
                raise CVGError(msg)
        else:
            msg = 'heidel_welch_number_points = '
            msg += '{} '.format(heidel_welch_number_points)
            msg += 'is the number of points and should be a positive `int`.'
            raise CVGError(msg)

        self.heidel_welch_k = heidel_welch_number_points
        self.heidel_welch_n = heidel_welch_number_points * 4
        self.heidel_welch_p = confidence_coefficient

        # Auxiliary matrix
        aux_array = np.arange(1, self.heidel_welch_k + 1) * 4 - 1.0
        aux_array /= (2.0 * self.heidel_welch_n)

        self.a_matrix = np.empty((self.heidel_welch_k, 4), dtype=np.float64)

        self.a_matrix[:, 0] = np.ones((self.heidel_welch_k), dtype=np.float64)
        self.a_matrix[:, 1] = aux_array
        self.a_matrix[:, 2] = aux_array * aux_array
        self.a_matrix[:, 3] = self.a_matrix[:, 2] * aux_array

        # The (Moore-Penrose) pseudo-inverse of a matrix.
        # Calculate the generalized inverse of a matrix using
        # its singular-value decomposition (SVD) and including all
        # large singular values.
        self.a_matrix_1_inv = pinv(self.a_matrix[:, :2])
        self.a_matrix_2_inv = pinv(self.a_matrix[:, :3])
        self.a_matrix_3_inv = pinv(self.a_matrix)

        # Heidelberger and Welch (1981) constants Table 1
        _sigma2 = 0.645 * \
            inv(np.dot(np.transpose(self.a_matrix[:, :2]),
                       self.a_matrix[:, :2]))[0, 0]
        self.heidel_welch_c1_1 = np.exp(-_sigma2 / 2.)
        # Heidelberger and Welch's C2 constant for
        # the first degree polynomial fit.
        self.heidel_welch_c2_1 = int(np.rint(2. / (np.exp(_sigma2) - 1.)))

        _sigma2 = 0.645 * \
            inv(np.dot(np.transpose(self.a_matrix[:, :3]),
                       self.a_matrix[:, :3]))[0, 0]
        self.heidel_welch_c1_2 = np.exp(-_sigma2 / 2.)
        # Heidelberger and Welch's C2 constant for
        # the second degree polynomial fit.
        self.heidel_welch_c2_2 = int(np.rint(2. / (np.exp(_sigma2) - 1.)))

        _sigma2 = 0.645 * \
            inv(np.dot(np.transpose(self.a_matrix), self.a_matrix))[0, 0]
        self.heidel_welch_c1_3 = np.exp(-_sigma2 / 2.)
        # Heidelberger and Welch's C2 constant for
        # the third degree polynomial fit.
        self.heidel_welch_c2_3 = int(np.rint(2. / (np.exp(_sigma2) - 1.)))

        self.tm_1 = t_inv_cdf(confidence_coefficient, self.heidel_welch_c2_1)
        self.tm_2 = t_inv_cdf(confidence_coefficient, self.heidel_welch_c2_2)
        self.tm_3 = t_inv_cdf(confidence_coefficient, self.heidel_welch_c2_3)

        # Set the flag
        self.heidel_welch_set = True

    def unset_heidel_welch_constants(self):
        """Unset the Heidelberger and Welch flag."""
        # Unset the flag
        self.heidel_welch_set = False
        self.heidel_welch_k = None
        self.heidel_welch_n = None
        self.heidel_welch_p = None
        self.a_matrix = None
        self.a_matrix_1_inv = None
        self.a_matrix_2_inv = None
        self.a_matrix_3_inv = None
        self.heidel_welch_c1_1 = None
        self.heidel_welch_c1_2 = None
        self.heidel_welch_c1_3 = None
        self.heidel_welch_c2_1 = None
        self.heidel_welch_c2_2 = None
        self.heidel_welch_c2_3 = None
        self.tm_1 = None
        self.tm_2 = None
        self.tm_3 = None

    def get_heidel_welch_constants(self):
        """Get the Heidelberger and Welch constants."""
        return \
            self.heidel_welch_set, \
            self.heidel_welch_k, \
            self.heidel_welch_n, \
            self.heidel_welch_p, \
            self.a_matrix, \
            self.a_matrix_1_inv, \
            self.a_matrix_2_inv, \
            self.a_matrix_3_inv, \
            self.heidel_welch_c1_1, \
            self.heidel_welch_c1_2, \
            self.heidel_welch_c1_3, \
            self.heidel_welch_c2_1, \
            self.heidel_welch_c2_2, \
            self.heidel_welch_c2_3, \
            self.tm_1, \
            self.tm_2, \
            self.tm_3

    def is_heidel_welch_set(self):
        """Return `True` if the flag is set to `True`."""
        return self.heidel_welch_set

    def get_heidel_welch_knp(self):
        """Get the heidel_welch_number_points, n, and confidence_coefficient constants."""
        return \
            self.heidel_welch_k, \
            self.heidel_welch_n, \
            self.heidel_welch_p

    def get_heidel_welch_auxilary_matrices(self):
        """Get the Heidelberger and Welch auxilary matrices."""
        return \
            self.a_matrix, \
            self.a_matrix_1_inv, \
            self.a_matrix_2_inv, \
            self.a_matrix_3_inv

    def get_heidel_welch_c1(self):
        """Get the Heidelberger and Welch C1 constants."""
        return \
            self.heidel_welch_c1_1, \
            self.heidel_welch_c1_2, \
            self.heidel_welch_c1_3

    def get_heidel_welch_c2(self):
        """Get the Heidelberger and Welch C2 constants."""
        return \
            self.heidel_welch_c2_1, \
            self.heidel_welch_c2_2, \
            self.heidel_welch_c2_3

    def get_heidel_welch_tm(self):
        """Get the Heidelberger and Welch t_distribution ppf.

        Get the Heidelberger and Welch t_distribution ppf for C2 degrees of
        freedom.
        """
        return self.tm_1, self.tm_2, self.tm_3


def ucl(time_series_data, *,
        confidence_coefficient=0.95,
        heidel_welch_number_points=50,
        fft=True,
        test_size=None,
        train_size=None,
        heidel_welch=None):
    r"""Approximate the upper confidence limit of the mean.

    Approximate an unbiased estimate of the upper confidence limit or
    half the width of the `confidence_coefficient%` probability interval
    (confidence interval, or credible interval) around the time-series mean.

    An estimate of the variance of the time-series mean is obtained by
    estimating the spectral density at zero frequency [12]_. We use an
    adaptive method which select the degree of the polynomial according to
    the shape of the modified periodogram [2]_.

    The estimated halfwidth of the confidence interval of time-series mean
    is computed as :math:`\frac{UCL}{\hat{\mu}}.`
    Where, UCL is the upper confidence limit, and :math:`\hat{\mu}` is the
    time-series mean.

    The upper confidence limit can be computed as,

    .. math::

        UCL = t_m\left(\text{confidence_coefficient}\right)\left(\hat{P}(0)/N\right)^{1/2},

    where :math:`N` is the number of data points, and :math:`t` is a
    t-distribution with :math:`m=C_2` degrees of freedom.

    For :math:`\text{confidence_coefficient} = 0.95`, or with the chance about
    95% (and 10 degrees of freedom) the t-value is 1.812. It implies that the
    true mean is lying between :math:`\hat{\mu} \pm UCL` with chance about 95%.
    In other words, the probability that the true mean lies between the upper
    and lower threshold is 95%.

    It should also be pointed out that as the sample size increases, a UCL of
    the mean approaches (converges to) the population mean.

    Args:
        time_series_data {array_like, 1d}: time series data.

    Keyword Args:
        confidence_coefficient (float, optional): probability (or confidence
            interval) and must be between 0.0 and 1.0, and represents the
            confidence for calculation of relative halfwidths estimation.
            (default: 0.95)
        heidel_welch_number_points (int, optional): the number of points that
            are used to obtain the polynomial fit. The parameter
            ``heidel_welch_number_points`` determines the frequency range
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
        heidel_welch (obj, optional): An instance of the HeidelbergerWelch
          object. (default: None)

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
    time_series_data = np.array(time_series_data, copy=False)

    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    if heidel_welch is None:
        hwl = HeidelbergerWelch(
            confidence_coefficient=confidence_coefficient,
            heidel_welch_number_points=heidel_welch_number_points)
    else:
        hwl = heidel_welch

        # We compute once and use it during iterations
        if not hwl.heidel_welch_set or \
            heidel_welch_number_points != hwl.heidel_welch_k or \
                confidence_coefficient != hwl.heidel_welch_p:
            hwl.set_heidel_welch_constants(
                confidence_coefficient=confidence_coefficient,
                heidel_welch_number_points=heidel_welch_number_points)

    batch_size = time_series_data.size // hwl.heidel_welch_n

    if batch_size < 1:
        msg = 'not enough data points (batching of the data '
        msg += 'is not possible).\nThe input time series has '
        msg += '{} data points which is '.format(time_series_data.size)
        msg += 'smaller than the minimum number of required points = '
        msg += '{} for batching.'.format(hwl.heidel_welch_n)
        raise CVGError(msg)

    # Batch the data
    x_batch = batch(time_series_data,
                    batch_size=batch_size,
                    with_centering=False,
                    with_scaling=False)

    n_batches = x_batch.size

    if n_batches != hwl.heidel_welch_n:
        if n_batches <= hwl.heidel_welch_n:
            msg = 'batching of the time series failed. (or '
            msg += 'there is not enough data points)\n'
            msg += 'Number of batches = {} '.format(n_batches)
            msg += 'must be the same as {}.'.format(hwl.heidel_welch_n)
            raise CVGError(msg)

        n_batches = hwl.heidel_welch_n
        x_batch = x_batch[:n_batches]

    # Compute the periodogram of the sequence x_batch
    period = modified_periodogram(x_batch,
                                  fft=(n_batches > 30 and fft),
                                  with_mean=False)

    left_range = range(0, period.size, 2)
    right_range = range(1, period.size, 2)

    # Compute the log of the average of adjacent periodogram values
    avg_period_lg = period[left_range] + period[right_range]
    avg_period_lg *= 0.5
    avg_period_lg = np.log(avg_period_lg)
    avg_period_lg += 0.27

    # Using ordinary least squares, and fit a polynomial to the data

    if test_size is None and train_size is None:
        # Least-squares solution
        least_sqr_sol_1 = np.matmul(hwl.a_matrix_1_inv, avg_period_lg)
        # Error of solution ||avg_period_lg - a_matrix*time_series_data||
        eps1 = norm(avg_period_lg -
                    np.matmul(hwl.a_matrix[:, :2], least_sqr_sol_1))

        # Least-squares solution
        least_sqr_sol_2 = np.matmul(hwl.a_matrix_2_inv, avg_period_lg)
        # Error of solution ||avg_period_lg - a_matrix*time_series_data||
        eps2 = norm(avg_period_lg -
                    np.matmul(hwl.a_matrix[:, :3], least_sqr_sol_2))

        # Least-squares solution
        least_sqr_sol_3 = np.matmul(hwl.a_matrix_3_inv, avg_period_lg)
        # Error of solution ||avg_period_lg - a_matrix*time_series_data||
        eps3 = norm(avg_period_lg - np.matmul(hwl.a_matrix, least_sqr_sol_3))
    else:
        ind_train, ind_test = train_test_split(
            avg_period_lg, train_size=train_size, test_size=test_size)

        # Least-squares solution
        least_sqr_sol_1 = np.matmul(
            hwl.a_matrix_1_inv[:, ind_train], avg_period_lg[ind_train])
        # Error of solution ||avg_period_lg - a_matrix*time_series_data||
        eps1 = norm(avg_period_lg[ind_test] -
                    np.matmul(hwl.a_matrix[ind_test, :2], least_sqr_sol_1))

        # Least-squares solution
        least_sqr_sol_2 = np.matmul(
            hwl.a_matrix_2_inv[:, ind_train], avg_period_lg[ind_train])
        # Error of solution ||avg_period_lg - a_matrix*time_series_data||
        eps2 = norm(avg_period_lg[ind_test] -
                    np.matmul(hwl.a_matrix[ind_test, :3], least_sqr_sol_2))

        # Least-squares solution
        least_sqr_sol_3 = np.matmul(
            hwl.a_matrix_3_inv[:, ind_train], avg_period_lg[ind_train])
        # Error of solution ||avg_period_lg - a_matrix*time_series_data||
        eps3 = norm(avg_period_lg[ind_test] -
                    np.matmul(hwl.a_matrix[ind_test, :], least_sqr_sol_3))

    # Find the best fit
    best_fit_index = np.argmin((eps1, eps2, eps3))

    if best_fit_index == 0:
        # get unbiased_estimate, which is an unbiased estimate of
        # log(confidence_coefficient(0)).
        unbiased_estimate = least_sqr_sol_1[0]
        heidel_welch_c = hwl.heidel_welch_c1_1
        hwl_tm = hwl.tm_1
    elif best_fit_index == 1:
        # get unbiased_estimate, which is an unbiased estimate of
        # log(confidence_coefficient(0)).
        unbiased_estimate = least_sqr_sol_2[0]
        heidel_welch_c = hwl.heidel_welch_c1_2
        hwl_tm = hwl.tm_2
    else:
        # get unbiased_estimate, which is an unbiased estimate of
        # log(confidence_coefficient(0)).
        unbiased_estimate = least_sqr_sol_3[0]
        heidel_welch_c = hwl.heidel_welch_c1_3
        hwl_tm = hwl.tm_3

    # The variance of the sample mean of a covariance stationary sequence is
    # given approximately by confidence_coefficient(O)/N, the spectral density
    # at zero frequency divided by the sample size.

    # Calculate the approximately unbiased estimate of the variance of the
    # sample mean
    sigma_sq = heidel_welch_c * np.exp(unbiased_estimate) / float(n_batches)

    # The standard deviation of the mean within the dataset. The
    # standard_error_of_mean provides a measurement for spread. The smaller
    # the spread the more accurate.
    standard_error_of_mean = np.sqrt(sigma_sq)

    upper_confidence_limit = hwl_tm * standard_error_of_mean
    return upper_confidence_limit


def subsamples_ucl(time_series_data, *,
                   confidence_coefficient=0.95,
                   population_standard_deviation=None,
                   subsample_indices=None,
                   si=None,
                   fft=True,
                   minimum_correlation_time=None):
    """Approximate the upper confidence limit of the mean.

    - If the population standard deviation is known, and
      `population_standard_deviation` is given,

      .. math::

            UCL = t_{\alpha,d} \left(\frac{\text population\ standard\ deviation}{\sqrt{n}}\right)

    - If the population standard deviation is unknown, the sample standard
      deviation is estimated and be used as `sample_standard_deviation`,

      .. math::

            UCL = t_{\alpha,d} \left(\frac{\text sample\ standard\ deviation}{\sqrt{n}}\right)

    In both cases, the ``Student's t`` distribution is used as the critical
    value. This value depends on the `confidence_coefficient` and the degrees
    of freedom, which is found by subtracting one from the number of
    observations.

    Args:
        time_series_data {array_like, 1d}: time series data.

    Keyword Args:
        confidence_coefficient (float, optional): probability (or confidence
            interval) and must be between 0.0 and 1.0, and represents the
            confidence for calculation of relative halfwidths estimation.
            (default: 0.95)
        population_standard_deviation (float, optional): population standard
            deviation. (default: None)
        subsample_indices (array_like, 1d, optional): indices of uncorrelated
            subsamples of the time series data. (default: None)
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool): if True, use FFT convolution. FFT should be preferred
            for long time series. (default: False)
        minimum_correlation_time (int, optional): minimum amount of correlation
            function to compute. The algorithm terminates after computing the
            correlation time out to minimum_correlation_time when the
            correlation function first goes negative. (default: None)

    Returns:
        float: upper_confidence_limit
            The approximately unbiased estimate of variance of the sample mean.

    """
    time_series_data = np.array(time_series_data, copy=False)

    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    if subsample_indices is None:
        subsamples_size = time_series_data.size
        if subsamples_size < 11:
            if subsamples_size < 5:
                msg = 'Number of samples = {}. '.format(subsamples_size)
                msg = 'UCL is restricted to have at least 5 samples.'
                raise CVGError(msg)
            subsample_indices = np.arange(subsamples_size)
        else:
            try:
                subsample_indices = subsample_index(
                    time_series_data, si=si, fft=fft,
                    minimum_correlation_time=minimum_correlation_time)
            except CVGError:
                msg = 'Failed to compute the indices of uncorrelated '
                msg += 'subsamples of the time_series_data.'
                raise CVGError(msg)

    else:
        subsample_indices = np.array(subsample_indices, copy=False)

        if subsample_indices.ndim != 1:
            msg = 'subsample_indices is not an array of one-dimension.'
            raise CVGError(msg)

    if subsample_indices.size < 5:
        if subsample_indices.size < 2:
            msg = 'Number of samples = {}. '.format(subsample_indices.size)
            msg += 'There are not enough samples.'
            raise CVGError(msg)

        msg = 'Number of samples = {}.\n'.format(subsample_indices.size)
        msg = 'UCL is restricted to have at least 5 samples.'
        cvg_warning(msg)

    try:
        uncorrelated_subsamples = time_series_data[subsample_indices]
    except IndexError:
        wrong_indices = np.where(subsample_indices >= time_series_data.size)
        msg = "Index = " if len(wrong_indices[0]) == 1 else "Indices = "
        msg += ",".join(map(str, subsample_indices[wrong_indices]))
        msg += " is out " if len(wrong_indices[0]) == 1 else " are out of "
        msg += "bound for " if len(wrong_indices[0]) == 1 else "bounds for "
        msg += "time_series_data with size {}".format(time_series_data.size)
        raise CVGError(msg)

    # Degrees of freedom
    uncorrelated_subsamples_size = uncorrelated_subsamples.size

    # If the population standard deviation is unknown
    if population_standard_deviation is None:
        # Compute the sample standard deviation
        sample_standard_deviation = np.std(uncorrelated_subsamples)

        # Compute the standard deviation of the mean within the dataset. The
        # standard_error_of_mean provides a measurement for spread. The smaller
        # the spread the more accurate.
        standard_error_of_mean = \
            sample_standard_deviation / np.sqrt(uncorrelated_subsamples_size)
    # If the population standard deviation is known
    else:
        # Compute the standard deviation of the mean within the dataset. The
        # standard_error_of_mean provides a measurement for spread. The smaller
        # the spread the more accurate.
        standard_error_of_mean = \
            population_standard_deviation /
        np.sqrt(uncorrelated_subsamples_size)

    # Compute the t_distribution confidence interval. When using the
    # t-distribution to compute a confidence interval, df = n - 1.
    coeff = t_inv_cdf(confidence_coefficient, uncorrelated_subsamples_size - 1)

    upper_confidence_limit = coeff * standard_error_of_mean
    return upper_confidence_limit


ucl_methods = {
    'heidel_welch': ucl,
    'subsample': subsamples_ucl,
}
