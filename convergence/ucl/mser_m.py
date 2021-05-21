"""MSER-m UCL module."""

from math import isclose, sqrt
import numpy as np

from .ucl_base import UCLBase
from convergence import \
    batch, \
    CVGError, \
    t_inv_cdf
from convergence._default import \
    _DEFAULT_ABS_TOL, \
    _DEFAULT_CONFIDENCE_COEFFICIENT, \
    _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE, \
    _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS, \
    _DEFAULT_BATCH_SIZE, \
    _DEFAULT_FFT, \
    _DEFAULT_SCALE_METHOD, \
    _DEFAULT_WITH_CENTERING, \
    _DEFAULT_WITH_SCALING, \
    _DEFAULT_TEST_SIZE, \
    _DEFAULT_TRAIN_SIZE, \
    _DEFAULT_POPULATION_STANDARD_DEVIATION, \
    _DEFAULT_SI, \
    _DEFAULT_MINIMUM_CORRELATION_TIME, \
    _DEFAULT_UNCORRELATED_SAMPLE_INDICES, \
    _DEFAULT_SAMPLE_METHOD, \
    _DEFAULT_IGNORE_END, \
    _DEFAULT_NSKIP, \
    _DEFAULT_NUMBER_OF_CORES


__all__ = [
    'mser_m',
    'MSER_m',
    'mser_m_ucl',
    'mser_m_ci',
    'mser_m_relative_half_width_estimate',
]


def mser_m(time_series_data,
           *,
           batch_size=_DEFAULT_BATCH_SIZE,
           scale=_DEFAULT_SCALE_METHOD,
           with_centering=_DEFAULT_WITH_CENTERING,
           with_scaling=_DEFAULT_WITH_SCALING,
           ignore_end=_DEFAULT_IGNORE_END):
    r"""Determine the truncation point using marginal standard error rules.

    Determine the truncation point using marginal standard error rules
    (MSER). The MSER [3]_ and MSER-5 [4]_ rules determine the truncation
    point as the value of :math:`d` that best balances the tradeoff between
    improved accuracy (elimination of bias) and decreased precision
    (reduction in the sample size) for the input series. They select a
    truncation point that minimizes the width of the marginal confidence
    interval about the truncated sample mean. The marginal confidence
    interval is a measure of the homogeneity of the truncated series.
    The optimal truncation point :math:`d(j)^*` selected by MSER-m can be
    expressed as:

    .. math::

        d(j)^* = \underset{n>d(j) \geq 0}{\text{argmin}}\left[\frac{1}{(n(j)-d(j))^2} \sum_{i=d}^{n}{\left(X_i(j)- \bar{X}_{n,d}(j) \right )^2}\right]

    MSER-m applies the equation to a series of batch averages instead of the
    raw series.

    Args:
        time_series_data (array_like, 1d): Time series data.
        batch_size (int, optional): batch size. (default: 5)
        scale (str, optional): A method to standardize a dataset.
            (default: 'translate_scale')
        with_centering (bool, optional): If True, use time_series_data minus the scale metod
            centering approach. (default: False)
        with_scaling (bool, optional): If True, scale the data to scale metod
            scaling approach. (default: False)
        ignore_end (int, or float, or None, optional): if `int`, it is
            the last few batch points that should be ignored. if `float`,
            should be in `(0, 1)` and it is the percent of last batch points
            that should be ignored. if `None` it would be set to the
            :math:`Min(batch_size, number_batches / 4)`. (default: None)

    Returns:
        bool, int: truncated, truncation point.
            Truncation point is the index to truncate.

    Note:
        MSER-m sometimes erroneously reports a truncation point at the end of
        the data series. This is because the method can be overly sensitive to
        observations at the end of the data series that are close in value.
        Here, we avoid this artifact, by not allowing the algorithm to consider
        the standard errors calculated from the last few data points.

    Note:
        If the truncation point returned by MSER-m > n/2, it is considered an
        invalid value and `truncated` will return as `False`. It means the
        method has not been provided with enough data to produce a valid
        result, and more data is required.

    Note:
        If the truncation obtained by MSER-m is the last index of the batched
        data, the MSER-m returns the time series data's last index as the
        truncation point. This index can be used as a measure that the
        algorithm did not find any truncation point.

    """
    time_series_data = np.array(time_series_data, copy=False)

    # Check inputs
    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    # Special case if timeseries is constant.
    _std = np.std(time_series_data)

    if not np.isfinite(_std):
        msg = 'there is at least one value in the input array which is '
        msg += 'non-finite or not-number.'
        raise CVGError(msg)

    if isclose(_std, 0, abs_tol=_DEFAULT_ABS_TOL):
        if not isinstance(batch_size, int):
            msg = 'batch_size = {} is not an `int`.'.format(batch_size)
            raise CVGError(msg)

        if batch_size < 1:
            msg = 'batch_size = {} < 1 is not valid.'.format(batch_size)
            raise CVGError(msg)

        if time_series_data.size < batch_size:
            return False, 0

        return True, 0

    del _std

    # Initialize
    x_batch = batch(time_series_data,
                    batch_size=batch_size,
                    scale=scale,
                    with_centering=with_centering,
                    with_scaling=with_scaling)

    # Number of batches
    number_batches = x_batch.size

    if not isinstance(ignore_end, int):
        if ignore_end is None:
            ignore_end = max(1, batch_size)
            ignore_end = min(ignore_end, number_batches // 4)
        elif isinstance(ignore_end, float):
            if not 0.0 < ignore_end < 1.0:
                msg = 'invalid ignore_end = {}. If '.format(ignore_end)
                msg += 'ignore_end input is a `float`, it should be in a '
                msg += '`(0, 1)` range.'
                raise CVGError(msg)
            ignore_end *= number_batches
            ignore_end = max(1, int(ignore_end))
        else:
            msg = 'invalid ignore_end = {}. '.format(ignore_end)
            msg += 'ignore_end is not an `int`, `float`, or `None`.'
            raise CVGError(msg)
    elif ignore_end < 1:
        msg = 'invalid ignore_end = {}. '.format(ignore_end)
        msg += 'ignore_end should be a positive `int`.'
        raise CVGError(msg)

    if number_batches <= ignore_end:
        msg = 'invalid ignore_end = {}.\n'.format(ignore_end)
        msg += 'Wrong number of batches is requested to be ignored '
        msg += 'from the total {} batches.'.format(number_batches)
        raise CVGError(msg)

    # To find the optimal truncation point in MSER-m

    number_batches_minus_d_inv = 1. / np.arange(number_batches, 0, -1)

    x_batch_sum = np.add.accumulate(x_batch[::-1])[::-1]
    x_batch_sum_sq = x_batch_sum * x_batch_sum
    x_batch_sum_sq *= number_batches_minus_d_inv

    number_batches_minus_d_inv *= number_batches_minus_d_inv

    x_batch_sq = x_batch * x_batch
    x_batch_sq_sum = np.add.accumulate(x_batch_sq[::-1])[::-1]

    d = number_batches_minus_d_inv * (x_batch_sq_sum - x_batch_sum_sq)

    # Convert truncation from batch to raw data
    truncate_index = np.nanargmin(d[:-ignore_end]) * batch_size

    # Correct the size of data
    processed_sample_size = number_batches * batch_size

    # Any truncation value > processed_sample_size / 2
    # is considered an invalid value and rejected
    if truncate_index > processed_sample_size // 2:
        # If the truncate_index is the last element of the batched data,
        # do the correction and return the last index of the
        # time_series_data array
        ignore_end += 1
        if truncate_index == (processed_sample_size - ignore_end * batch_size):
            truncate_index = time_series_data.size

        return False, truncate_index

    return True, truncate_index


class MSER_m(UCLBase):
    r"""MSER-m algorithm.

    The MSER [3]_ and MSER-5 [4]_ rules determine the truncation point as the
    value of :math:`d` that best balances the tradeoff between improved
    accuracy (elimination of bias) and decreased precision (reduction in the
    sample size) for the input series. They select a truncation point that
    minimizes the width of the marginal confidence interval about the truncated
    sample mean. The marginal confidence interval is a measure of the
    homogeneity of the truncated series.
    The optimal truncation point :math:`d(j)^*` selected by MSER-m can be
    expressed as:

    .. math::

        d(j)^* = \underset{n>d(j) \geq 0}{\text{argmin}}\left[\frac{1}{(n(j)-d(j))^2} \sum_{i=d}^{n}{\left(X_i(j)- \bar{X}_{n,d}(j) \right )^2}\right]

    MSER-m applies the equation to a series of batch averages instead of the
    raw series. The CI estimators can be computed from the truncated sequence
    of batch means.

    References:
        .. [3] White, K.P., Jr., (1997). "An effective truncation heuristic
               for bias reduction in simulation output.". Simulation.,
               69(6), p. 323--334.
        .. [4] Spratt, S. C., (1998). "Heuristics for the startup problem."
               M.S. Thesis, Department OS Systems Engineering, University
               of Virginia.

    """

    def __init__(self):
        self.name = 'mser_m'

        UCLBase.__init__(self)

    def estimate_equilibration_length(
        self,
        time_series_data,
        *,
        batch_size=_DEFAULT_BATCH_SIZE,
        scale=_DEFAULT_SCALE_METHOD,
        with_centering=_DEFAULT_WITH_CENTERING,
        with_scaling=_DEFAULT_WITH_SCALING,
        ignore_end=_DEFAULT_IGNORE_END,
        # unused input parmeters in MSER-m UCL module
        # estimate_equilibration_length interface
        si=_DEFAULT_SI,
        nskip=_DEFAULT_NSKIP,
        fft=_DEFAULT_FFT,
        minimum_correlation_time=_DEFAULT_MINIMUM_CORRELATION_TIME,
        number_of_cores=_DEFAULT_NUMBER_OF_CORES):
        """Estimate the equilibration point in a time series data."""
        truncated, truncate_index = mser_m(
            time_series_data=time_series_data,
            batch_size=batch_size,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
            ignore_end=ignore_end)

        if truncated:
            time_series_data = np.array(time_series_data, copy=False)

            self.set_si(
                time_series_data=time_series_data[truncate_index:],
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time)

            return True, truncate_index

        self.si = None
        return False, truncate_index

    def ucl(self,
            time_series_data,
            *,
            confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
            batch_size=_DEFAULT_BATCH_SIZE,
            scale=_DEFAULT_SCALE_METHOD,
            with_centering=_DEFAULT_WITH_CENTERING,
            with_scaling=_DEFAULT_WITH_SCALING,
            # unused input parmeters in
            # MSER_m ucl interface
            equilibration_length_estimate=_DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
            heidel_welch_number_points=_DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
            fft=_DEFAULT_FFT,
            test_size=_DEFAULT_TEST_SIZE,
            train_size=_DEFAULT_TRAIN_SIZE,
            population_standard_deviation=_DEFAULT_POPULATION_STANDARD_DEVIATION,
            si=_DEFAULT_SI,
            minimum_correlation_time=_DEFAULT_MINIMUM_CORRELATION_TIME,
            uncorrelated_sample_indices=_DEFAULT_UNCORRELATED_SAMPLE_INDICES,
            sample_method=_DEFAULT_SAMPLE_METHOD):
        r"""Approximate the upper confidence limit of the mean [20]_.

        Args:
            time_series_data (array_like, 1d): time series data.
            confidence_coefficient (float, optional): probability (or confidence
                interval) and must be between 0.0 and 1.0, and represents the
                confidence for calculation of relative halfwidths estimation.
                (default: 0.95)
            batch_size (int, optional): batch size. (default: 5)
            scale (str, optional): A method to standardize a dataset.
                (default: 'translate_scale)
            with_centering (bool, optional): If True, use time_series_data
                minus the scale metod centering approach. (default: False)
            with_scaling (bool, optional): If True, scale the data to scale
                metod scaling approach. (default: False)

        Returns:
            float: upper_confidence_limit

        References:
            .. [20] Mokashi, A. C. and Tejada, J. J. and Yousefi, S. and
                    Tafazzoli, A. and Xu, T. and Wilson, J. R. and Steiger,
                    N. M., (2010). "Performance comparison of MSER-5 and
                    N-Skart on the simulation start-up problem," Proceedings of
                    the 2010 Winter Simulation Conference, p. 971--982.
                    doi = 10.1109/WSC.2010.5679094

        """
        time_series_data = np.array(time_series_data, copy=False)

        if time_series_data.ndim != 1:
            msg = 'time_series_data is not an array of one-dimension.'
            raise CVGError(msg)

        if confidence_coefficient <= 0.0 or confidence_coefficient >= 1.0:
            msg = 'confidence_coefficient = {} '.format(confidence_coefficient)
            msg += 'is not in the range (0.0 1.0).'
            raise CVGError(msg)

        # Initialize
        x_batch = batch(time_series_data,
                        batch_size=batch_size,
                        scale=scale,
                        with_centering=with_centering,
                        with_scaling=with_scaling)

        # Number of batches
        number_batches = x_batch.size

        # compute and set the mean (grand average of the truncated batch means)
        self.mean = x_batch.mean()

        # compute and set the sample standard deviation (sample variance of the
        # truncated batch means)
        self.std = x_batch.std()
        self.sample_size = number_batches

        # Compute the standard deviation of the mean within the dataset. The
        # standard_error_of_mean provides a measurement for spread. The smaller
        # the spread the more accurate. Please see ref [20]_
        standard_error_of_mean = self.std / sqrt(number_batches)

        # Compute the t_distribution confidence interval. When using the
        # t-distribution to compute a confidence interval, df = n - 1.
        p_up = (1 + confidence_coefficient) / 2
        # Please see ref [20]_
        upper = t_inv_cdf(p_up, number_batches - 1)

        upper_confidence_limit = upper * standard_error_of_mean
        return upper_confidence_limit


def mser_m_ucl(time_series_data,
               *,
               confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
               batch_size=_DEFAULT_BATCH_SIZE,
               scale=_DEFAULT_SCALE_METHOD,
               with_centering=_DEFAULT_WITH_CENTERING,
               with_scaling=_DEFAULT_WITH_SCALING,
               obj=None):
    """Approximate the upper confidence limit of the mean."""
    mser = MSER_m() if obj is None else obj
    upper_confidence_limit = mser.ucl(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        batch_size=batch_size,
        scale=scale,
        with_centering=with_centering,
        with_scaling=with_scaling)
    return upper_confidence_limit


def mser_m_ci(time_series_data,
              *,
              confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
              batch_size=_DEFAULT_BATCH_SIZE,
              scale=_DEFAULT_SCALE_METHOD,
              with_centering=_DEFAULT_WITH_CENTERING,
              with_scaling=_DEFAULT_WITH_SCALING,
              obj=None):
    r"""Approximate the confidence interval of the mean [20]_.

    Args:
        time_series_data (array_like, 1d): time series data.
        confidence_coefficient (float, optional): probability (or confidence
            interval) and must be between 0.0 and 1.0, and represents the
            confidence for calculation of relative halfwidths estimation.
            (default: 0.95)
        batch_size (int, optional): batch size. (default: 5)
        scale (str, optional): A method to standardize a dataset.
            (default: 'translate_scale)
        with_centering (bool, optional): If True, use time_series_data
            minus the scale metod centering approach. (default: False)
        with_scaling (bool, optional): If True, scale the data to scale
            metod scaling approach. (default: False)
        obj (MSER_m, optional): instance of ``MSER_m`` (default: None)

    Returns:
        float, float: confidence interval

    """
    mser = MSER_m() if obj is None else obj
    confidence_limits = mser.ci(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        batch_size=batch_size,
        scale=scale,
        with_centering=with_centering,
        with_scaling=with_scaling)
    return confidence_limits


def mser_m_relative_half_width_estimate(
        time_series_data,
        *,
        confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
        batch_size=_DEFAULT_BATCH_SIZE,
        scale=_DEFAULT_SCALE_METHOD,
        with_centering=_DEFAULT_WITH_CENTERING,
        with_scaling=_DEFAULT_WITH_SCALING,
        obj=None):
    r"""Get the relative half width estimate.

    The relative half width estimate is the confidence interval
    half-width or upper confidence limit (UCL) divided by the sample mean.

    The UCL is calculated as a `confidence_coefficient%` confidence
    interval for the mean, using the portion of the time series data, which
    is in the stationarity region.

    Args:
        time_series_data (array_like, 1d): time series data.
        confidence_coefficient (float, optional): probability (or confidence
            interval) and must be between 0.0 and 1.0, and represents the
            confidence for calculation of relative halfwidths estimation.
            (default: 0.95)
        batch_size (int, optional): batch size. (default: 5)
        scale (str, optional): A method to standardize a dataset.
            (default: 'translate_scale)
        with_centering (bool, optional): If True, use time_series_data
            minus the scale metod centering approach. (default: False)
        with_scaling (bool, optional): If True, scale the data to scale
            metod scaling approach. (default: False)
        obj (MSER_m, optional): instance of ``MSER_m`` (default: None)

    Returns:
        float: the relative half width estimate.

    """
    mser = MSER_m() if obj is None else obj
    try:
        relative_half_width_estimate = \
            mser.relative_half_width_estimate(
                time_series_data=time_series_data,
                confidence_coefficient=confidence_coefficient,
                batch_size=batch_size,
                scale=scale,
                with_centering=with_centering,
                with_scaling=with_scaling)
    except CVGError:
        raise CVGError('Failed to get the relative_half_width_estimate.')
    return relative_half_width_estimate
