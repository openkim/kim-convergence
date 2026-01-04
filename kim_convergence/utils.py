r"""Utility module."""

import numpy as np
from typing import Union

from ._default import _DEFAULT_TEST_SIZE, _DEFAULT_TRAIN_SIZE, _DEFAULT_SEED
from kim_convergence import CRError


__all__ = ["train_test_split", "validate_split"]


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

    # ensure at least one size is given
    if test_size is None and train_size is None:
        if default_test_size is None:
            raise CRError(
                "test_size, train_size, and default_test_size can not be "
                "`None` at the same time."
            )
        test_size = default_test_size

    def _check(name: str, val: Union[int, float]) -> None:
        if isinstance(val, int):
            if val <= 0 or val >= n_samples:
                raise CRError(f"{name}={val} must be in (0, {n_samples}).")
        elif isinstance(val, float):
            if val <= 0.0 or val >= 1.0:
                raise CRError(f"{name}={val} must be in (0, 1).")
        else:
            raise CRError(f"{name} has invalid type {type(val)}.")

    if test_size is not None:
        _check("test_size", test_size)

    if train_size is not None:
        _check("train_size", train_size)

    if (
        isinstance(test_size, float)
        and isinstance(train_size, float)
        and test_size + train_size > 1.0
    ):
        raise CRError("Sum of fractional sizes > 1.")

    if isinstance(test_size, float):
        n_test = int(np.ceil(test_size * n_samples))
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        n_test = None

    if isinstance(train_size, float):
        n_train = int(np.floor(train_size * n_samples))
    elif isinstance(train_size, int):
        n_train = train_size
    else:
        n_train = None

    if train_size is None:  # test_size is authoritative
        assert isinstance(n_test, int)  # keeps mypy happy
        n_train = n_samples - n_test
        if n_train <= 0:
            raise CRError("Derived train set is empty.")
    elif test_size is None:  # train_size is authoritative
        assert isinstance(n_train, int)  # keeps mypy happy
        n_test = n_samples - n_train
        if n_test <= 0:
            raise CRError("Derived test set is empty.")

    assert isinstance(n_test, int)  # keeps mypy happy
    assert isinstance(n_train, int)  # keeps mypy happy
    if n_train + n_test > n_samples:
        raise CRError(
            f"train_size ({n_train}) + test_size ({n_test}) > "
            f"n_samples ({n_samples})."
        )
    if n_train == 0:
        raise CRError("The resulting train set is empty.")
    if n_test == 0:
        raise CRError("The resulting test set is empty.")

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
    ind_train = permutation[n_test:(n_test + n_train)]
    return ind_train, ind_test
