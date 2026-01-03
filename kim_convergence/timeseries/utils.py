r"""Time series utility module."""

import numpy as np
from random import randint
from typing import Optional, Union

from .statistical_inefficiency import si_methods

from kim_convergence import CRError
from kim_convergence._default import (
    _DEFAULT_SI,
    _DEFAULT_FFT,
    _DEFAULT_MINIMUM_CORRELATION_TIME,
    _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
    _DEFAULT_SAMPLE_METHOD,
)

__all__ = [
    "time_series_data_si",
    "uncorrelated_time_series_data_sample_indices",
    "uncorrelated_time_series_data_samples",
    "time_series_data_uncorrelated_samples",
    "time_series_data_uncorrelated_random_samples",
    "time_series_data_uncorrelated_block_averaged_samples",
]

SAMPLING_METHODS = ("uncorrelated", "random", "block_averaged")


def _out_of_bounds_error_str(bad_indices: list[int], size: int) -> str:
    r"""
    Return a uniform, ready-to-raise error message for out-of-bounds indices.
    Caller must `raise CRError(...)` explicitly.
    """
    n_bad = len(bad_indices)
    prefix = "Index" if n_bad == 1 else "Indices"
    verb = "is" if n_bad == 1 else "are"
    bounds = "bound" if n_bad == 1 else "bounds"

    idx_str = ", ".join(map(str, bad_indices))
    return (
        f"{prefix} = {{{idx_str}}} {verb} out of {bounds} "
        f"for time_series_data with size of {size}."
    )


def time_series_data_si(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
) -> float:
    r"""Helper method to compute or return the statistical inefficiency value.

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
        si = "statistical_inefficiency"

    if isinstance(si, str):
        if si not in si_methods:
            raise CRError(
                f"method {si} not found. Valid statistical inefficiency (si) "
                "methods are:\n\t- " + "\n\t- ".join(si_methods)
            )

        si_func = si_methods[si]

        try:
            si_value = si_func(
                time_series_data,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
            )
        except CRError as e:
            raise CRError(
                "Failed to compute the statistical inefficiency for the "
                "time_series_data."
            ) from e

    elif isinstance(si, (float, int)):
        if si < 1.0:
            raise CRError(
                f"statistical inefficiency = {si} must be greater than or " "equal one."
            )

        si_value = si

    else:
        raise CRError("statistical inefficiency (si) must be a `float` or a `str`.")

    return si_value


def uncorrelated_time_series_data_sample_indices(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
) -> np.ndarray:
    r"""Return indices of uncorrelated subsamples of the time series data.

    Return indices of the uncorrelated uncorrelated_sample of the time series data.
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
        minimum_correlation_time=minimum_correlation_time,
    )

    # Get the length of the time_series_data
    time_series_data_size = len(time_series_data)

    uncorrelated_sample_indices = si_value * np.arange(time_series_data_size)

    # Each block should contain more steps than si
    uncorrelated_sample_indices = np.ceil(uncorrelated_sample_indices).astype(int)

    indices = np.where(uncorrelated_sample_indices < time_series_data_size)

    # Assemble list of indices of uncorrelated snapshots return it.
    return uncorrelated_sample_indices[indices]


def uncorrelated_time_series_data_samples(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
    uncorrelated_sample_indices: Union[
        np.ndarray, list[int], None
    ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
    sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD,
) -> np.ndarray:
    r"""Get time series data at the sample_method uncorrelated_sample indices.

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
        1darray: uncorrelated_sample of the time series data.
            time series data at uncorrelated uncorrelated_sample indices.

    """
    if sample_method is None:
        sample_method = "uncorrelated"

    if not isinstance(sample_method, str):
        raise CRError(f"sample_method {sample_method} is not a `str`.")

    if sample_method not in SAMPLING_METHODS:
        raise CRError(
            f"method {sample_method} not found. Valid sampling methods "
            "are:\n\t- " + "\n\t- ".join(SAMPLING_METHODS)
        )

    if sample_method == "uncorrelated":
        return time_series_data_uncorrelated_samples(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            uncorrelated_sample_indices=uncorrelated_sample_indices,
        )

    if sample_method == "random":
        return time_series_data_uncorrelated_random_samples(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            uncorrelated_sample_indices=uncorrelated_sample_indices,
        )

    return time_series_data_uncorrelated_block_averaged_samples(
        time_series_data=time_series_data,
        si=si,
        fft=fft,
        minimum_correlation_time=minimum_correlation_time,
        uncorrelated_sample_indices=uncorrelated_sample_indices,
    )


def time_series_data_uncorrelated_samples(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
    uncorrelated_sample_indices: Union[
        np.ndarray, list[int], None
    ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
) -> np.ndarray:
    r"""Return time series data at uncorrelated uncorrelated_sample indices.

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

    Returns:
        1darray: uncorrelated_sample of the time series data.
            time series data at uncorrelated uncorrelated_sample indices.

    """
    time_series_data = np.asarray(time_series_data)

    # Check inputs
    if time_series_data.ndim != 1:
        raise CRError("time_series_data is not an array of one-dimension.")

    if uncorrelated_sample_indices is None:
        try:
            indices = uncorrelated_time_series_data_sample_indices(
                time_series_data=time_series_data,
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
            )
        except CRError as e:
            raise CRError(
                "Failed to compute the indices of uncorrelated samples of"
                "the time_series_data."
            ) from e

    else:
        indices = np.asarray(uncorrelated_sample_indices)

        if indices.ndim != 1:
            raise CRError(
                "uncorrelated_sample_indices is not an array of one-dimension."
            )

    try:
        uncorrelated_samples = time_series_data[indices]
    except IndexError as e:
        time_series_data_size = time_series_data.size
        mask = indices >= time_series_data_size
        wrong_indices = np.where(mask)
        raise CRError(
            _out_of_bounds_error_str(
                bad_indices=indices[wrong_indices], size=time_series_data_size  # type: ignore[arg-type]
            )
        ) from e

    return uncorrelated_samples


def time_series_data_uncorrelated_random_samples(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
    uncorrelated_sample_indices: Union[
        np.ndarray, list[int], None
    ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
) -> np.ndarray:
    r"""Return random data for each block after blocking the data.

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
        1darray: uncorrelated_sample of the time series data.
            random data for each block after blocking the time series data.

    """
    time_series_data = np.asarray(time_series_data)

    # Check inputs
    if time_series_data.ndim != 1:
        raise CRError("time_series_data is not an array of one-dimension.")

    if uncorrelated_sample_indices is None:
        try:
            indices = uncorrelated_time_series_data_sample_indices(
                time_series_data=time_series_data,
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
            )
        except CRError as e:
            raise CRError(
                "Failed to compute the indices of uncorrelated samples of "
                "the time_series_data."
            ) from e

    else:
        indices = np.asarray(uncorrelated_sample_indices)

        if indices.ndim != 1:
            raise CRError(
                "uncorrelated_sample_indices is not an array of one-dimension."
            )

    time_series_data_size = time_series_data.size

    wrong_indices = np.where(indices >= time_series_data_size)

    if len(wrong_indices[0]) > 0:
        raise CRError(
            _out_of_bounds_error_str(
                bad_indices=indices[wrong_indices], size=time_series_data_size  # type: ignore[arg-type]
            )
        )

    random_samples = np.empty(indices.size - 1, dtype=time_series_data.dtype)

    index_s = indices[0]
    for index, index_e in enumerate(indices[1:]):
        rand_index = randint(index_s, index_e - 1)
        random_samples[index] = time_series_data[rand_index]
        index_s = index_e

    return random_samples


def time_series_data_uncorrelated_block_averaged_samples(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
    uncorrelated_sample_indices: Union[
        np.ndarray, list[int], None
    ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
) -> np.ndarray:
    """Return average value for each block after blocking the data.

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
        1darray: uncorrelated_sample of the time series data.
            average value for each block after blocking the time series
            data.

    """
    time_series_data = np.asarray(time_series_data)

    # Check inputs
    if time_series_data.ndim != 1:
        raise CRError("time_series_data is not an array of one-dimension.")

    if uncorrelated_sample_indices is None:
        try:
            indices = uncorrelated_time_series_data_sample_indices(
                time_series_data=time_series_data,
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
            )
        except CRError as e:
            raise CRError(
                "Failed to compute the indices of uncorrelated samples of "
                "the time_series_data."
            ) from e

    else:
        indices = np.asarray(uncorrelated_sample_indices)

        if indices.ndim != 1:
            raise CRError(
                "uncorrelated_sample_indices is not an array of one-dimension."
            )

    time_series_data_size = time_series_data.size

    wrong_indices = np.where(indices >= time_series_data_size)

    if len(wrong_indices[0]) > 0:
        raise CRError(
            _out_of_bounds_error_str(
                bad_indices=indices[wrong_indices], size=time_series_data_size  # type: ignore[arg-type]
            )
        )

    block_averaged_samples = np.empty(indices.size - 1, dtype=time_series_data.dtype)

    index_s = 0
    for index, index_e in enumerate(indices[1:]):
        block_averaged_samples[index] = np.mean(time_series_data[index_s:index_e])
        index_s = index_e

    return block_averaged_samples
