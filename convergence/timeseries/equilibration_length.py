r"""Time series equilibration length estimation module.

This module uses a conceptually simple automated procedure developed by Chodera
[11]_ that does not make strict assumptions about the distribution of the
observable of interest. The equilibration is chosen to maximize the number of
uncorrelated samples.

"""

from math import isclose
import numpy as np

from convergence import \
    CVGError, \
    statistical_inefficiency,\
    r_statistical_inefficiency, \
    split_r_statistical_inefficiency, \
    split_statistical_inefficiency, \
    si_methods
from convergence._default import \
    __ABS_TOL, \
    __SI, \
    __FFT, \
    __MINIMUM_CORRELATION_TIME, \
    __IGNORE_END, \
    __NSKIP, \
    __BATCH_SIZE, \
    __SCALE_METHOD, \
    __WITH_CENTERING, \
    __WITH_SCALING

__WITH_SCALING, \


__all__ = [
    'estimate_equilibration_length',
]


def estimate_equilibration_length(
        time_series_data,
        *,
        si=__SI,
        nskip=__NSKIP,
        fft=__FFT,
        minimum_correlation_time=__MINIMUM_CORRELATION_TIME,
        ignore_end=__IGNORE_END,
        # unused input parmeters in Time series module
        # estimate_equilibration_length interface
        batch_size=__BATCH_SIZE,
        scale=__SCALE_METHOD,
        with_centering=__WITH_CENTERING,
        with_scaling=__WITH_SCALING):
    """Estimate the equilibration point in a time series data.

    Estimate the equilibration point in a time series data using the
    statistical inefficiencies [11]_, [8]_, [9]_.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (str, optional): statistical inefficiency method. (default: None)
        nskip (int, optional): the number of data points to skip.
            (default: 1)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): the minimum amount of
            correlation function to compute. The algorithm terminates after
            computing the correlation time out to minimum_correlation_time when
            the correlation function first goes negative. (default: None)
        ignore_end (int, or float, or None, optional): if ``int``, it is the
            last few points that should be ignored. if ``float``, should be in
            ``(0, 1)`` and it is the percent of number of points that should be
            ignored. If ``None`` it would be set to the one fourth of the total
            number of points. (default: None)

    Returns:
        int, float: equilibration index, statistical inefficiency estimates
            equilibration index, and statitical inefficiency estimates of a
            time series at the equilibration index estimate.

    References:
        .. [11] Chodera, J. D., (2016). "A Simple Method for Automated
                Equilibration Detection in Molecular Simulations". J. Chem.
                Theory and Comp., Simulation., 12(4), p. 1799--1805.

    """
    time_series_data = np.array(time_series_data, copy=False)

    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    # Get the length of the timeseries.
    time_series_data_size = time_series_data.size

    if isinstance(si, str):
        if si not in si_methods:
            msg = 'method {} not found. Valid statistical '.format(si)
            msg += 'inefficiency (si) methods are:\n\t- '
            msg += '{}'.format('\n\t- '.join(si_methods))
            raise CVGError(msg)
    elif si is None:
        si = 'statistical_inefficiency'
    else:
        msg = 'si is not a `str` or `None`.'
        raise CVGError(msg)

    si_func = si_methods[si]

    if not isinstance(nskip, int):
        if nskip is not None:
            msg = 'nskip must be an `int`.'
            raise CVGError(msg)

        nskip = 1

    elif nskip < 1:
        msg = 'nskip must be a positive `int`.'
        raise CVGError(msg)

    if not isinstance(ignore_end, int):
        if ignore_end is None:
            ignore_end = max(1, time_series_data_size // 4)
        elif isinstance(ignore_end, float):
            if not 0.0 < ignore_end < 1.0:
                msg = 'invalid ignore_end = {}. If '.format(ignore_end)
                msg += 'ignore_end input is a `float`, it should be in a '
                msg += '`(0, 1)` range.'
                raise CVGError(msg)
            ignore_end *= time_series_data_size
            ignore_end = max(1, int(ignore_end))
        else:
            msg = 'invalid ignore_end = {}. '.format(ignore_end)
            msg += 'ignore_end is not an `int`, `float`, or `None`.'
            raise CVGError(msg)
    elif ignore_end < 1:
        msg = 'invalid ignore_end = {}. '.format(ignore_end)
        msg += 'ignore_end should be a positive `int`.'
        raise CVGError(msg)

    # Upper bound check
    if si == 'r_statistical_inefficiency':
        if time_series_data_size < 4:
            msg = '{} input data points are not '.format(time_series_data_size)
            msg += 'sufficient to be used by "{}".'.format(si)
            raise CVGError(msg)
        ignore_end = max(3, ignore_end)
    elif si == 'split_r_statistical_inefficiency':
        if time_series_data_size < 8:
            msg = '{} input data points are not '.format(time_series_data_size)
            msg += 'sufficient to be used by "{}".'.format(si)
            raise CVGError(msg)
        ignore_end = max(7, ignore_end)
    elif si == 'split_statistical_inefficiency':
        if time_series_data_size < 8:
            msg = '{} input data points are not '.format(time_series_data_size)
            msg += 'sufficient to be used by "{}".'.format(si)
            raise CVGError(msg)
        ignore_end = max(7, ignore_end)

    if time_series_data_size <= ignore_end:
        msg = 'invalid ignore_end = {}.\n'.format(ignore_end)
        msg = 'Wrong number of data points is requested to be '
        msg += 'ignored from {} total points.'.format(time_series_data_size)
        raise CVGError(msg)

    # Special case if timeseries is constant.
    _std = time_series_data.std()

    if not np.isfinite(_std):
        msg = 'there is at least one value in the input '
        msg += 'array which is non-finite or not-number.'
        raise CVGError(msg)

    if isclose(_std, 0, abs_tol=__ABS_TOL):
        # index and si
        return 0, time_series_data_size

    del _std

    # Upper bound check
    upper_bound = time_series_data_size - ignore_end

    nskip = min(nskip, upper_bound)

    # Estimate of statistical inefficiency
    statistical_inefficiency_estimate = 1.0

    # Effective samples size
    effective_samples_size = 0.0

    # Equilibration estimate index
    equilibration_index_estimate = 0

    # Compute the statitical inefficiency of a time series
    for t in range(0, upper_bound, nskip):
        # slice a numpy array, the memory is shared between the
        # slice and the original
        x = time_series_data[t:]

        x_size = float(x.size)

        try:
            si_value = si_func(
                x, fft=fft, minimum_correlation_time=minimum_correlation_time)
        except CVGError:
            si_value = x_size

        effective_samples_size_ = x_size / si_value

        # Find the maximum
        if effective_samples_size_ > effective_samples_size:
            statistical_inefficiency_estimate = si_value
            effective_samples_size = effective_samples_size_
            equilibration_index_estimate = t

    return equilibration_index_estimate, statistical_inefficiency_estimate
