"""Time series utility module."""

import numpy as np
from random import randint

from .statistical_inefficiency import \
    statistical_inefficiency, \
    r_statistical_inefficiency, \
    split_r_statistical_inefficiency, \
    split_statistical_inefficiency, \
    si_methods

from convergence import CVGError
from convergence._default import \
    __SI, \
    __FFT, \
    __MINIMUM_CORRELATION_TIME, \
    __UNCORRELATED_SAMPLE_INDICES, \
    __SAMPLE_METHOD

__all__ = [
    'time_series_data_si',
    'uncorrelated_time_series_data_sample_indices',
    'uncorrelated_time_series_data_samples',
    'time_series_data_uncorrelated_samples',
    'time_series_data_uncorrelated_random_samples',
    'time_series_data_uncorrelated_block_averaged_samples',
]

SAMPLING_METHODS = ('uncorrelated', 'random', 'block_averaged')


def time_series_data_si(time_series_data,
                        *,
                        si=__SI,
                        fft=__FFT,
                        minimum_correlation_time=__MINIMUM_CORRELATION_TIME):
    """Helper method to compute or return the statistical inefficiency value.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool, optional): if True, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of
            correlation function to compute. The algorithm terminates after
            computing the correlation time out to minimum_correlation_time
            when the correlation function first goes negative.
            (default: None)

    Returns:
        float: estimated statistical inefficiency value.
            :math:`si >= 1` is the estimated statistical inefficiency.

    """
    if si is None:
        si = 'statistical_inefficiency'

    if isinstance(si, str):
        if si not in si_methods:
            msg = 'method {} not found. Valid statistical '.format(si)
            msg += 'inefficiency (si) methods are:\n\t- '
            msg += '{}'.format('\n\t- '.join(si_methods))
            raise CVGError(msg)

        si_func = si_methods[si]

        try:
            si_value = si_func(
                time_series_data,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time)
        except CVGError:
            msg = 'Failed to compute the statistical inefficiency '
            msg += 'for the time_series_data.'
            raise CVGError(msg)

    elif isinstance(si, (float, int)):
        if si < 1.0:
            msg = 'statistical inefficiency = {} must be '.format(si)
            msg += 'greater than or equal one.'
            raise CVGError(msg)

        si_value = si

    else:
        msg = 'statistical inefficiency (si) must be '
        msg += 'a `float` or a `str`.'
        raise CVGError(msg)

    return si_value


def uncorrelated_time_series_data_sample_indices(
        time_series_data,
        *,
        si=__SI,
        fft=__FFT,
        minimum_correlation_time=__MINIMUM_CORRELATION_TIME):
    r"""Return indices of uncorrelated subsamples of the time series data.

    Return indices of the uncorrelated subsample of the time series data.
    Subsample a correlated timeseries to extract an effectively
    uncorrelated dataset. If si (statistical inefficiency) is not provided
    it will be computed.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool, optional): if True, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of
            correlation function to compute. The algorithm terminates after
            computing the correlation time out to minimum_correlation_time
            when the correlation function first goes negative.
            (default: None)

    Returns:
        1darray: indices array.
            Indices of uncorrelated subsamples of the time series data.

    """
    si_value = time_series_data_si(
        time_series_data,
        si=si,
        fft=fft,
        minimum_correlation_time=minimum_correlation_time)

    # Get the length of the time_series_data
    time_series_data_size = len(time_series_data)

    uncorrelated_sample_indices = si_value * np.arange(time_series_data_size)

    # Each block should contain more steps than si
    uncorrelated_sample_indices = \
        np.ceil(uncorrelated_sample_indices).astype(int)

    indices = np.where(uncorrelated_sample_indices < time_series_data_size)

    # Assemble list of indices of uncorrelated snapshots return it.
    return uncorrelated_sample_indices[indices]


def uncorrelated_time_series_data_samples(
        time_series_data,
        *,
        si=__SI,
        fft=__FFT,
        minimum_correlation_time=__MINIMUM_CORRELATION_TIME,
        uncorrelated_sample_indices=__UNCORRELATED_SAMPLE_INDICES,
        sample_method=__SAMPLE_METHOD):
    r"""Get time series data at the sample_method subsample indices.

    Subsample a correlated timeseries to extract an effectively uncorrelated
    dataset. If si (statistical inefficiency) is not provided it will be
    computed.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
        fft (bool, optional): if True, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of
            correlation function to compute. The algorithm terminates after
            computing the correlation time out to minimum_correlation_time
            when the correlation function first goes negative.
            (default: None)
        uncorrelated_sample_indices (array_like, 1d, optional): indices
            of uncorrelated subsamples of the time series data.
            (default: None)
        sample_method (str, optional): sampling method, one of the
            ``uncorrelated``, ``random``, or ``block_averaged``.
            (default: None)

    Returns:
        1darray: subsample of the time series data.
            time series data at uncorrelated subsample indices.

    """
    if sample_method is None:
        sample_method = 'uncorrelated'

    if not isinstance(sample_method, str):
        msg = 'sample_method {} is not a `str`.'.format(sample_method)
        raise CVGError(msg)

    if sample_method not in SAMPLING_METHODS:
        msg = 'method {} not found. Valid '.format(sample_method)
        msg += 'sampling methods are:\n\t- '
        msg += '{}'.format('\n\t- '.join(SAMPLING_METHODS))
        raise CVGError(msg)

    if sample_method == 'uncorrelated':
        return time_series_data_uncorrelated_samples(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            uncorrelated_sample_indices=uncorrelated_sample_indices)

    if sample_method == 'random':
        return time_series_data_uncorrelated_random_samples(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            uncorrelated_sample_indices=uncorrelated_sample_indices)

    return time_series_data_uncorrelated_block_averaged_samples(
        time_series_data=time_series_data,
        si=si,
        fft=fft,
        minimum_correlation_time=minimum_correlation_time,
        uncorrelated_sample_indices=uncorrelated_sample_indices)


def time_series_data_uncorrelated_samples(
        time_series_data,
        *,
        si=__SI,
        fft=__FFT,
        minimum_correlation_time=__MINIMUM_CORRELATION_TIME,
        uncorrelated_sample_indices=__UNCORRELATED_SAMPLE_INDICES):
    r"""Return time series data at uncorrelated subsample indices.

    Subsample a correlated timeseries to extract an effectively uncorrelated
    dataset. If si (statistical inefficiency) is not provided it will be
    computed.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
            c
        fft (bool, optional): if True, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of
            correlation function to compute. The algorithm terminates after
            computing the correlation time out to minimum_correlation_time
            when the correlation function first goes negative.
            (default: None)
        uncorrelated_sample_indices (array_like, 1d, optional): indices
            of uncorrelated subsamples of the time series data.
            (default: None)
        uncorrelated_sample_indices (array_like, 1d, optional): indices
            of uncorrelated subsamples of the time series data.
            (default: None)

    Returns:
        1darray: subsample of the time series data.
            time series data at uncorrelated subsample indices.

    """
    time_series_data = np.array(time_series_data, copy=False)

    # Check inputs
    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    if uncorrelated_sample_indices is None:
        try:
            indices = uncorrelated_time_series_data_sample_indices(
                time_series_data=time_series_data,
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time)
        except CVGError:
            msg = 'Failed to compute the indices of uncorrelated '
            msg += 'samples of the time_series_data.'
            raise CVGError(msg)

    else:
        indices = np.array(uncorrelated_sample_indices, copy=False)

        if indices.ndim != 1:
            msg = 'uncorrelated_sample_indices is not '
            msg += 'an array of one-dimension.'
            raise CVGError(msg)

    try:
        uncorrelated_samples = time_series_data[indices]
    except IndexError:
        time_series_data_size = time_series_data.size
        mask = indices >= time_series_data_size
        wrong_indices = np.where(mask)
        msg = 'Index = {' if len(wrong_indices[0]) == 1 else 'Indices = {'
        msg += ', '.join(map(str, indices[wrong_indices]))
        msg += '} is out ' if len(wrong_indices[0]) == 1 else '} are out '
        msg += 'of bound ' if len(wrong_indices[0]) == 1 else 'of bounds '
        msg += 'for time_series_data with size of '
        msg += "{}".format(time_series_data_size)
        raise CVGError(msg)

    return uncorrelated_samples


def time_series_data_uncorrelated_random_samples(
        time_series_data,
        *,
        si=__SI,
        fft=__FFT,
        minimum_correlation_time=__MINIMUM_CORRELATION_TIME,
        uncorrelated_sample_indices=__UNCORRELATED_SAMPLE_INDICES):
    r"""Retuen random data for each block after blocking the data.

    At first, break down the time series data into the series of blocks,
    where each block contains ``si`` successive data points. If si
    (statistical inefficiency) is not provided it will be computed. Then a
    single value is taken at random from each block.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool, optional): if True, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of
            correlation function to compute. The algorithm terminates after
            computing the correlation time out to minimum_correlation_time
            when the correlation function first goes negative.
            (default: None)
        uncorrelated_sample_indices (array_like, 1d, optional): indices
            of uncorrelated subsamples of the time series data.
            (default: None)

    Returns:
        1darray: subsample of the time series data.
            random data for each block after blocking the time series data.

    """
    time_series_data = np.array(time_series_data, copy=False)

    # Check inputs
    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    if uncorrelated_sample_indices is None:
        try:
            indices = uncorrelated_time_series_data_sample_indices(
                time_series_data=time_series_data,
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time)
        except CVGError:
            msg = 'Failed to compute the indices of uncorrelated '
            msg += 'samples of the time_series_data.'
            raise CVGError(msg)

    else:
        indices = np.array(uncorrelated_sample_indices, copy=False)

        if indices.ndim != 1:
            msg = 'uncorrelated_sample_indices is not '
            msg += 'an array of one-dimension.'
            raise CVGError(msg)

    time_series_data_size = time_series_data.size

    wrong_indices = np.where(indices >= time_series_data_size)

    if len(wrong_indices[0]) > 0:
        msg = 'Index = {' if len(wrong_indices[0]) == 1 else 'Indices = {'
        msg += ', '.join(map(str, indices[wrong_indices]))
        msg += '} is out ' if len(wrong_indices[0]) == 1 else '} are out '
        msg += 'of bound ' if len(wrong_indices[0]) == 1 else 'of bounds '
        msg += 'for time_series_data with size of '
        msg += "{}".format(time_series_data_size)
        raise CVGError(msg)

    random_samples = np.empty(indices.size, dtype=time_series_data.dtype)

    index_s = 0
    for index, index_e in enumerate(indices[1:-1]):
        rand_index = randint(index_s, index_e - 1)
        random_samples[index] = time_series_data[rand_index]
        index_s = index_e

    rand_index = randint(index_s, indices[-1])
    random_samples[-1] = time_series_data[rand_index]

    return random_samples


def time_series_data_uncorrelated_block_averaged_samples(
        time_series_data,
        *,
        si=__SI,
        fft=__FFT,
        minimum_correlation_time=__MINIMUM_CORRELATION_TIME,
        uncorrelated_sample_indices=__UNCORRELATED_SAMPLE_INDICES):
    """Retuen average value for each block after blocking the data.

    At first, break down the time series data into the series of blocks,
    where each block contains ``si`` successive data points. If si
    (statistical inefficiency) is not provided it will be computed. Then
    the average value for each block is determined. This coarse graining
    approach is commonly used for thermodynamic properties.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool, optional): if True, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of
            correlation function to compute. The algorithm terminates after
            computing the correlation time out to minimum_correlation_time
            when the correlation function first goes negative.
            (default: None)
        uncorrelated_sample_indices (array_like, 1d, optional): indices
            of uncorrelated subsamples of the time series data.
            (default: None)

    Returns:
        1darray: subsample of the time series data.
            average value for each block after blocking the time series
            data.

    """
    time_series_data = np.array(time_series_data, copy=False)

    # Check inputs
    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    if uncorrelated_sample_indices is None:
        try:
            indices = uncorrelated_time_series_data_sample_indices(
                time_series_data=time_series_data,
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time)
        except CVGError:
            msg = 'Failed to compute the indices of uncorrelated '
            msg += 'samples of the time_series_data.'
            raise CVGError(msg)

    else:
        indices = np.array(uncorrelated_sample_indices, copy=False)

        if indices.ndim != 1:
            msg = 'uncorrelated_sample_indices is not '
            msg += 'an array of one-dimension.'
            raise CVGError(msg)

    time_series_data_size = time_series_data.size

    wrong_indices = np.where(indices >= time_series_data_size)

    if len(wrong_indices[0]) > 0:
        msg = 'Index = {' if len(wrong_indices[0]) == 1 else 'Indices = {'
        msg += ', '.join(map(str, indices[wrong_indices]))
        msg += '} is out ' if len(wrong_indices[0]) == 1 else '} are out '
        msg += 'of bound ' if len(wrong_indices[0]) == 1 else 'of bounds '
        msg += 'for time_series_data with size of '
        msg += "{}".format(time_series_data_size)
        raise CVGError(msg)

    block_averaged_samples = \
        np.empty(indices.size - 1, dtype=time_series_data.dtype)

    index_s = 0
    for index, index_e in enumerate(indices[1:]):
        block_averaged_samples[index] = np.mean(
            time_series_data[index_s:index_e])
        index_s = index_e

    return block_averaged_samples
