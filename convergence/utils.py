"""Utility module."""

import numpy as np
from typing import Union, List

from ._default import \
    _DEFAULT_TEST_SIZE, \
    _DEFAULT_TRAIN_SIZE, \
    _DEFAULT_SEED
from convergence import CVGError


__all__ = [
    'validate_split',
    'train_test_split',
]


def validate_split(
    *,
    n_samples: int,
    train_size: Union[int, float, None],
    test_size: Union[int, float, None],
        default_test_size: Union[int, float, None] = None) -> tuple((int, int)):
    r"""Validate test/train sizes.

    Helper function to validate the test/train sizes to be meaningful with
    regard to the size of the data (n_samples)

    Args:
        n_samples (int): total number of sample points
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


def train_test_split(
        time_series_data: Union[np.ndarray, List[float]],
        *,
        train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
        test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
        seed: Union[int, np.random.RandomState, None] = _DEFAULT_SEED,
        default_test_size: Union[int, float, None] = 0.1) -> tuple((np.ndarray, np.ndarray)):
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
            will be set to ``default_test_size``. (default: 0.1)
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
