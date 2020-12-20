"""Utility module."""

import numpy as np
from random import randint

from .err import CVGError
from .batch import batch
from .statistical_inefficiency import \
    statistical_inefficiency, \
    r_statistical_inefficiency, \
    split_r_statistical_inefficiency, \
    split_statistical_inefficiency, \
    si_methods

__all__ = [
    'validate_split',
    'train_test_split',
    'subsample',
    'subsample_index',
    'random_subsample',
    'block_average_subsample',
]


def validate_split(*, n_samples, train_size, test_size, default_test_size=None):
    r"""Validate test/train sizes.

    Helper function to validate the test/train sizes to be meaningful with
    regard to the size of the data (n_samples)

    Args:
        n_samples (int): total number of sampl points
        train_size (int, float, or None): train size
        test_size (int, float, or None): test size
        default_test_size (int, float, or None, optional): default test size.
            (default: None)

    Returns:
        int, int: number of train points, number of test points

    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        msg = 'n_samples={} is not a positive '.format(n_samples)
        msg += '`int`.'
        raise CVGError(msg)

    if test_size is None and train_size is None:
        if default_test_size is None:
            msg = 'test_size, train_size, and default_test_size '
            msg += 'can not be `None` at the same time.'
            raise CVGError(msg)
        test_size = default_test_size

    test_size_int = isinstance(test_size, int)
    test_size_float = isinstance(test_size, float)
    train_size_int = isinstance(train_size, int)
    train_size_float = isinstance(train_size, float)

    if test_size_int and (test_size >= n_samples or test_size <= 0):
        msg = 'test_size={} should be positive '.format(test_size)
        msg += 'and smaller than the number of samples={}'.format(n_samples)
        raise CVGError(msg)
    
    if test_size_float and (test_size <= 0 or test_size >= 1):
        msg = 'test_size={} should be a float in '.format(test_size)
        msg += 'the [0, 1] range.'
        raise CVGError(msg)
    
    if test_size is not None:
        msg = 'Invalid input of test_size={}.'.format(test_size)
        raise CVGError(msg)

    if train_size_int and (train_size >= n_samples or train_size <= 0):
        msg = 'train_size={} should be positive '.format(train_size)
        msg += 'and smaller than the number of samples={}'.format(n_samples)
        raise CVGError(msg)

    if train_size_float and (train_size <= 0 or train_size >= 1):
        msg = 'train_size={} should be a float '.format(test_size)
        msg += 'in the [0, 1] range.'
        raise CVGError(msg)

    if train_size is not None:
        msg = 'Invalid input of train_size={}.'.format(train_size)
        raise CVGError(msg)

    if (test_size_float and train_size_float and train_size + test_size > 1):
        msg = 'The sum of test_size and train_size = '
        msg += '{}, should be in the '.format(train_size + test_size)
        msg += '[0, 1] range. Reduce test_size and/or train_size.'
        raise CVGError(msg)

    if test_size_float:
        n_test = np.ceil(test_size * n_samples)
    elif train_size_int:
        n_test = float(test_size)

    if train_size_float:
        n_train = np.floor(train_size * n_samples)
    elif train_size_int:
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        msg = 'the sum of train_size and test_size = '
        msg += '{}, should be smaller '.format(int(n_train + n_test))
        msg += 'than the number of samples {}.'.format(int(n_samples))
        msg += 'Reduce test_size and/or train_size.'
        raise CVGError(msg)

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        msg = 'the resulting train set is empty.'
        raise CVGError(msg)

    return n_train, n_test


def train_test_split(time_series_data, *,
                     train_size=None,
                     test_size=None,
                     seed=None,
                     default_test_size=0.1):
    r"""Split time_series_data into random train and test indices.

    Args:
        time_series_data (array_like): time series data, array-like of shape
            ``(n_samples, n_features)``, where n_samples is the number of
            samples and n_features is the number of features.
        test_size (int, float, or None, optional): if ``float``, should be
            between 0.0 and 1.0 and represent the proportion of the dataset to
            include in the test split. If ``int``, represents the absolute
            number of test samples. If ``None``, the value is set to the
            complement of the train size. If ``train_size`` is also None, it
            will be set to ``default_test_size``. (default: None)
        train_size (int, float, or None, optional): if ``float``, should be
            between 0.0 and 1.0 and represent the proportion of the dataset to
            include in the train split. If ``int``, represents the absolute
            number of train samples. If ``None``, the value is automatically
            set to the complement of the test size. (default: None)
        seed (None, int or `np.random.RandomState()`, optional): random number
            seed. (default: None)
        default_test_size (float, optional): Default test size. (default: 0.1)

    Returns:
        1darray, 1darray: training indices, testing indices.

    """
    time_series_data = np.array(time_series_data, copy=False)
    n_samples = np.shape(time_series_data)[0]
    n_train, n_test = validate_split(n_samples=n_samples,
                                     train_size=train_size,
                                     test_size=test_size,
                                     default_test_size=default_test_size)

    if seed is None or isinstance(seed, int):
        rng = np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        msg = 'seed should be one of `None`, `int` or '
        msg += '`np.random.RandomState`.'
        raise CVGError(msg)

    # random partition
    permutation = rng.permutation(n_samples)
    ind_test = permutation[:n_test]
    ind_train = permutation[n_test:(n_test + n_train)]
    return ind_train, ind_test


def subsample_index(time_series_data, *,
                    si=None,
                    fft=False,
                    minimum_correlation_time=None):
    r"""Return indices of uncorrelated subsamples of the time series data.

    Return indices of the uncorrelated subsample of the time series data.
    Subsample a correlated timeseries to extract an effectively uncorrelated
    dataset. If si (statistical inefficiency) is not provided it will be
    computed.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool): if True, use FFT convolution. FFT should be preferred
            for long time series. (default: False)
        minimum_correlation_time (int, optional): minimum amount of correlation 
            function to compute. The algorithm terminates after computing the 
            correlation time out to minimum_correlation_time when the 
            correlation function first goes negative. (default: None)

    Returns:
        1darray: indices array.
            Indices of uncorrelated subsamples of the time series data.

    """
    if si is None:
        try:
            # Compute the statistical inefficiency for the timeseries
            si = statistical_inefficiency(
                time_series_data, fft=fft,
                minimum_correlation_time=minimum_correlation_time)
        except CVGError:
            msg = 'Failed to compute the statistical inefficiency for the '
            msg += 'time_series_data'
            raise CVGError(msg)
    else:
        if isinstance(si, str):
            if si not in si_methods:
                msg = 'method {} not found. Valid statistical '.format(si)
                msg += 'inefficiency (si) methods are:\n\t- '
                msg += '{}'.format('\n\t- '.join(si_methods))
                raise CVGError(msg)

            si_func = si_methods[si]

            try:
                si = si_func(
                    time_series_data, fft=fft,
                    minimum_correlation_time=minimum_correlation_time)
            except CVGError:
                msg = 'Failed to compute the statistical inefficiency for the '
                msg += 'time_series_data'
                raise CVGError(msg)
        elif isinstance(si, (float, int)):
            if si < 1.0:
                msg = 'statistical inefficiency = {} must be '.format(si)
                msg += 'greater than or equal one.'
                raise CVGError(msg)
        else:
            msg = 'statistical inefficiency (si) must be a `float` or `str`.'
            raise CVGError(msg)

    # Get the length of the time_series_data
    n = len(time_series_data)

    _ind = si * np.arange(0, n)

    # Each block should contain more steps than si
    _ind = np.ceil(_ind).astype(int)
    _ind_indices = np.where(_ind < n)

    # Assemble list of indices of uncorrelated snapshots.
    indices = _ind[_ind_indices[0]]

    # Return the list of indices of uncorrelated snapshots.
    return indices


def subsample(time_series_data, *,
              si=None,
              fft=False,
              minimum_correlation_time=None):
    r"""Return time series data at uncorrelated subsample indices.

    Subsample a correlated timeseries to extract an effectively uncorrelated
    dataset. If si (statistical inefficiency) is not provided it will be
    computed.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool): if True, use FFT convolution. FFT should be preferred
            for long time series. (default: False)
        minimum_correlation_time (int, optional): minimum amount of correlation 
            function to compute. The algorithm terminates after computing the 
            correlation time out to minimum_correlation_time when the 
            correlation function first goes negative. (default: None)

    Returns:
        1darray: subsample of the time series data.
            time series data at uncorrelated subsample indices.

    """
    time_series_data = np.array(time_series_data, copy=False)

    # Check inputs
    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    try:
        indices = subsample_index(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time)
    except CVGError:
        msg = 'Failed to compute the indices of uncorrelated subsamples of '
        msg += 'the time_series_data.'
        raise CVGError(msg)

    return time_series_data[indices]


def random_subsample(time_series_data, *,
                     si=None,
                     fft=False,
                     minimum_correlation_time=None):
    r"""Retuen random data for each block after blocking the time series data.

    At first, break down the time series data into the series of blocks, where
    each block contains ``si`` successive data points. If si
    (statistical inefficiency) is not provided it will be computed. Then a
    single value is taken at random from each block.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool): if True, use FFT convolution. FFT should be preferred
            for long time series. (default: False)
        minimum_correlation_time (int, optional): minimum amount of correlation 
            function to compute. The algorithm terminates after computing the 
            correlation time out to minimum_correlation_time when the 
            correlation function first goes negative. (default: None)

    Returns:
        1darray: subsample of the time series data.
            random data for each block after blocking the time series data.

    """
    time_series_data = np.array(time_series_data, copy=False)

    # Check inputs
    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    try:
        indices = subsample_index(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time)
    except CVGError:
        msg = 'Failed to compute the indices of uncorrelated subsamples of '
        msg += 'the time_series_data.'
        raise CVGError(msg)

    random_subsample_data = np.empty(indices.size,
                                     dtype=time_series_data.dtype)

    index_s = 0
    for index, index_e in enumerate(indices[1:-1]):
        rand_index = randint(index_s, index_e - 1)
        random_subsample_data[index] = time_series_data[rand_index]
        index_s = index_e
    
    rand_index = randint(index_s, indices[-1])
    random_subsample_data[-1] = time_series_data[rand_index]

    return random_subsample_data


def block_average_subsample(time_series_data, *,
                            si=None,
                            fft=False,
                            minimum_correlation_time=None):
    """Retuen average value for each block after blocking the time series data.

    At first, break down the time series data into the series of blocks, where
    each block contains ``si`` successive data points. If si
    (statistical inefficiency) is not provided it will be computed. Then the
    average value for each block is determined. This coarse graining approach is
    commonly used for thermodynamic properties.

    Args:
        time_series_data (array_like, 1d): time series data.
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool): if True, use FFT convolution. FFT should be preferred
            for long time series. (default: False)
        minimum_correlation_time (int, optional): minimum amount of correlation 
            function to compute. The algorithm terminates after computing the 
            correlation time out to minimum_correlation_time when the 
            correlation function first goes negative. (default: None)

    Returns:
        1darray: subsample of the time series data.
            average value for each block after blocking the time series data.

    """
    time_series_data = np.array(time_series_data, copy=False)

    # Check inputs
    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    try:
        indices = subsample_index(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time)
    except CVGError:
        msg = 'Failed to compute the indices of uncorrelated subsamples of '
        msg += 'the time_series_data.'
        raise CVGError(msg)

    mean_subsample_data = np.empty(indices.size - 1,
                                   dtype=time_series_data.dtype)

    index_s = 0
    for index, index_e in enumerate(indices[1:]):
        mean_subsample_data[index] = np.mean(time_series_data[index_s:index_e])
        index_s = index_e

    return mean_subsample_data
