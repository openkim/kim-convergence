r"""Utility module."""

import numpy as np
from typing import Union

from ._default import _DEFAULT_TEST_SIZE, _DEFAULT_TRAIN_SIZE, _DEFAULT_SEED
from kim_convergence import CRError


__all__ = [
    "validate_split",
    "train_test_split",
]


def validate_split(
    *,
    n_samples: int,
    train_size: Union[int, float, None],
    test_size: Union[int, float, None],
    default_test_size: Union[int, float, None] = None,
) -> tuple[int, int]:
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
        raise CRError(f"n_samples={n_samples} is not a positive `int`.")

    if test_size is None and train_size is None:
        if default_test_size is None:
            raise CRError(
                "test_size, train_size, and default_test_size can not be "
                "`None` at the same time."
            )
        test_size = default_test_size

    test_size_int = isinstance(test_size, int)
    test_size_float = isinstance(test_size, float)
    train_size_int = isinstance(train_size, int)
    train_size_float = isinstance(train_size, float)

    if test_size_int and (test_size >= n_samples or test_size <= 0):
        raise CRError(
            f"test_size={test_size} should be positive and smaller than "
            f"the number of samples={n_samples}"
        )

    if test_size_float and (test_size <= 0 or test_size >= 1):
        raise CRError(f"test_size={test_size} should be a float in the [0, 1] range.")

    if test_size is not None:
        raise CRError(f"Invalid input of test_size={test_size}.")

    if train_size_int and (train_size >= n_samples or train_size <= 0):
        raise CRError(
            f"train_size={train_size} should be positive and smaller than "
            f"the number of samples={n_samples}"
        )

    if train_size_float and (train_size <= 0 or train_size >= 1):
        raise CRError(f"train_size={test_size} should be a float in the [0, 1] range.")

    if train_size is not None:
        raise CRError(f"Invalid input of train_size={train_size}.")

    if test_size_float and train_size_float and train_size + test_size > 1:
        raise CRError(
            "The sum of test_size and train_size = "
            f"{train_size + test_size}, should be in the "
            "[0, 1] range. Reduce test_size and/or train_size."
        )

    if test_size_float:
        n_test = np.ceil(test_size * n_samples)
    elif train_size_int:
        assert test_size is not None
        n_test = float(test_size)

    if train_size_float:
        n_train = np.floor(train_size * n_samples)
    elif train_size_int:
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:  # type: ignore[arg-type]
        raise CRError(
            f"the sum of train_size and test_size = {int(n_train + n_test)}, "  # type: ignore[arg-type]
            f"should be smaller than the number of samples {int(n_samples)}. "
            "Reduce test_size and/or train_size."
        )

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise CRError("the resulting train set is empty.")

    return n_train, n_test


def train_test_split(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
    test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
    seed: Union[int, np.random.RandomState, None] = _DEFAULT_SEED,
    default_test_size: Union[int, float, None] = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
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
    time_series_data = np.asarray(time_series_data)
    n_samples = np.shape(time_series_data)[0]
    n_train, n_test = validate_split(
        n_samples=n_samples,
        train_size=train_size,
        test_size=test_size,
        default_test_size=default_test_size,
    )

    if seed is None or isinstance(seed, int):
        rng = np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        raise CRError("seed should be one of `None`, `int` or `np.random.RandomState`.")

    # random partition
    permutation = rng.permutation(n_samples)
    ind_test = permutation[:n_test]
    ind_train = permutation[n_test : (n_test + n_train)]
    return ind_train, ind_test
