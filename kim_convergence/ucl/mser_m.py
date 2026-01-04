r"""MSER-m UCL module."""

from math import isclose, sqrt
import numpy as np
from typing import Optional, Union

from kim_convergence._default import (
    _DEFAULT_ABS_TOL,
    _DEFAULT_CONFIDENCE_COEFFICIENT,
    _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
    _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_FFT,
    _DEFAULT_SCALE_METHOD,
    _DEFAULT_WITH_CENTERING,
    _DEFAULT_WITH_SCALING,
    _DEFAULT_TEST_SIZE,
    _DEFAULT_TRAIN_SIZE,
    _DEFAULT_POPULATION_STANDARD_DEVIATION,
    _DEFAULT_SI,
    _DEFAULT_MINIMUM_CORRELATION_TIME,
    _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
    _DEFAULT_SAMPLE_METHOD,
    _DEFAULT_IGNORE_END,
    _DEFAULT_NSKIP,
    _DEFAULT_NUMBER_OF_CORES,
)
from .ucl_base import UCLBase
from kim_convergence import batch, CRError, CRSampleSizeError, t_inv_cdf


__all__ = [
    "mser_m",
    "MSER_m",
    "mser_m_ucl",
    "mser_m_ci",
    "mser_m_relative_half_width_estimate",
]


def _normalize_ignore_end(
    ignore_end: Union[int, float, None],
    batch_size: int,
    number_batches: int,
    data_size: int,
) -> int:
    r"""
    Validate *ignore_end* and return the positive number of batches to ignore.

    *None* is converted to ``max(1, batch_size, number_batches // 4)``.
    A *float* in ``(0, 1)`` is interpreted as a fraction of the total batches.
    An *int* must be ≥ 1.

    Returns
    -------
    int
        Positive number of batches to ignore from the end.

    Raises
    ------
    CRError
        If *ignore_end* has wrong type, wrong range, or is < 1.
    CRSampleSizeError
        If *ignore_end* >= *number_batches*.
    """

    if not isinstance(ignore_end, int):
        if ignore_end is None:
            ignore_end = max(1, batch_size)
            ignore_end = max(1, min(ignore_end, number_batches // 4))  # guaranteed >= 1
        elif isinstance(ignore_end, float):
            if not 0.0 < ignore_end < 1.0:
                raise CRError(
                    f"invalid ignore_end = {ignore_end}. If ignore_end input "
                    "is a `float`, it should be in a `(0, 1)` range."
                )
            ignore_end *= number_batches
            ignore_end = max(1, int(ignore_end))
        else:
            raise CRError(
                f"invalid ignore_end = {ignore_end}. ignore_end is not an "
                "`int`, `float`, or `None`."
            )

        if ignore_end < 1:
            raise CRError(
                "ignore_end is not given on input and it is automatically set = "
                f"{ignore_end} using {data_size} number of data points and the "
                f"batch size = {batch_size}.\nignore_end should be a positive `int`."
            )

    elif ignore_end < 1:
        raise CRError(
            f"invalid ignore_end = {ignore_end}. ignore_end should be a "
            "positive `int`."
        )

    if number_batches <= ignore_end:
        raise CRSampleSizeError(
            f"invalid ignore_end = {ignore_end}.\nWrong number of batches is "
            f"requested to be ignored from the total {number_batches} batches."
        )

    return ignore_end


def mser_m(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    scale: str = _DEFAULT_SCALE_METHOD,
    with_centering: bool = _DEFAULT_WITH_CENTERING,
    with_scaling: bool = _DEFAULT_WITH_SCALING,
    ignore_end: Union[int, float, None] = _DEFAULT_IGNORE_END,
) -> tuple[bool, int]:
    r"""Determine the truncation point using marginal standard error rules.

    Determine the truncation point using marginal standard error rules
    (MSER). The MSER [white1997]_ and MSER-5 [spratt1998]_ rules determine the
    truncation point as the value of :math:`d` that best balances the tradeoff
    between improved accuracy (elimination of bias) and decreased precision
    (reduction in the sample size) for the input series. They select a
    truncation point that minimizes the width of the marginal confidence
    interval about the truncated sample mean. The marginal confidence
    interval is a measure of the homogeneity of the truncated series.
    The optimal truncation point :math:`d(j)^*` selected by MSER-m can be
    expressed as:

    .. math::

        d(j)^* = \underset{n>d(j) \geq 0}{\text{argmin}}
        \left[
        \frac{1}{(n(j)-d(j))^2}
        \sum_{i=d}^{n}{\left(X_i(j)- \bar{X}_{n,d}(j) \right )^2}
        \right]

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
    time_series_data = np.asarray(time_series_data)

    # Check inputs
    if time_series_data.ndim != 1:
        raise CRError("time_series_data is not an array of one-dimension.")

    # Special case if timeseries is constant.
    _std = np.std(time_series_data)

    if not np.isfinite(_std):
        raise CRError(
            "there is at least one value in the input array which is "
            "non-finite or not-number."
        )

    if isclose(_std, 0, abs_tol=_DEFAULT_ABS_TOL):
        if not isinstance(batch_size, int):
            raise CRError(f"batch_size = {batch_size} is not an `int`.")

        if batch_size < 1:
            raise CRError(f"batch_size = {batch_size} < 1 is not valid.")

        if time_series_data.size < batch_size:
            return False, 0

        return True, 0

    del _std

    # Initialize
    x_batch = batch(
        time_series_data,
        batch_size=batch_size,
        scale=scale,
        with_centering=with_centering,
        with_scaling=with_scaling,
    )

    # Number of batches
    number_batches = x_batch.size

    ignore_end = _normalize_ignore_end(
        ignore_end, batch_size, number_batches, time_series_data.size
    )

    # To find the optimal truncation point in MSER-m

    number_batches_minus_d_inv = 1.0 / np.arange(number_batches, 0, -1)

    x_batch_sum = np.add.accumulate(x_batch[::-1])[::-1]
    x_batch_sum_sq = x_batch_sum * x_batch_sum
    x_batch_sum_sq *= number_batches_minus_d_inv

    number_batches_minus_d_inv *= number_batches_minus_d_inv

    x_batch_sq = x_batch * x_batch
    x_batch_sq_sum = np.add.accumulate(x_batch_sq[::-1])[::-1]

    d = number_batches_minus_d_inv * (x_batch_sq_sum - x_batch_sum_sq)

    # Convert truncation from batch to raw data
    truncate_index = np.nanargmin(d[:-ignore_end]) * batch_size
    # Convert from numpy.int64 to int
    truncate_index = int(truncate_index)

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

    The MSER [white1997]_ and MSER-5 [spratt1998]_ rules determine the
    truncation point as the value of :math:`d` that best balances the tradeoff
    between improved accuracy (elimination of bias) and decreased precision
    (reduction in the sample size) for the input series. They select a
    truncation point that minimizes the width of the marginal confidence
    interval about the truncated sample mean. The marginal confidence interval
    is a measure of the homogeneity of the truncated series.
    The optimal truncation point :math:`d(j)^*` selected by MSER-m can be
    expressed as:

    .. math::

        d(j)^* = \underset{n>d(j) \geq 0}{\text{argmin}}
        \left[
        \frac{1}{(n(j)-d(j))^2} \sum_{i=d}^{n}{\left(X_i(j)- \bar{X}_{n,d}(j) \right )^2}
        \right]

    MSER-m applies the equation to a series of batch averages instead of the
    raw series. The CI estimators can be computed from the truncated sequence
    of batch means.

    """

    def __init__(self):
        UCLBase.__init__(self)

        self.name = "mser_m"

    def estimate_equilibration_length(
        self,
        time_series_data: Union[np.ndarray, list[float]],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        scale: str = _DEFAULT_SCALE_METHOD,
        with_centering: bool = _DEFAULT_WITH_CENTERING,
        with_scaling: bool = _DEFAULT_WITH_SCALING,
        ignore_end: Union[int, float, None] = _DEFAULT_IGNORE_END,
        number_of_cores: int = _DEFAULT_NUMBER_OF_CORES,  # unused (API compatibility)
        si: Union[str, float, int, None] = _DEFAULT_SI,
        nskip: Optional[int] = _DEFAULT_NSKIP,  # unused (API compatibility)
        fft: bool = _DEFAULT_FFT,
        minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
    ) -> tuple[bool, int]:
        r"""Estimate the equilibration point in a time series data."""
        truncated, truncate_index = mser_m(
            time_series_data=time_series_data,
            batch_size=batch_size,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
            ignore_end=ignore_end,
        )

        if truncated:
            time_series_data = np.asarray(time_series_data)

            self.set_si(
                time_series_data=time_series_data[truncate_index:],
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
            )

            return True, truncate_index

        self.si = None
        return False, truncate_index

    def _ucl_impl(
        self,
        time_series_data: Union[np.ndarray, list[float]],
        *,
        confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        scale: str = _DEFAULT_SCALE_METHOD,
        with_centering: bool = _DEFAULT_WITH_CENTERING,
        with_scaling: bool = _DEFAULT_WITH_SCALING,
        equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,  # unused (API compatibility)
        heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,  # unused (API compatibility)
        fft: bool = _DEFAULT_FFT,  # unused (API compatibility)
        test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,  # unused (API compatibility)
        train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,  # unused (API compatibility)
        population_standard_deviation: Optional[
            float
        ] = _DEFAULT_POPULATION_STANDARD_DEVIATION,  # unused (API compatibility)
        si: Union[str, float, int, None] = _DEFAULT_SI,  # unused (API compatibility)
        minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,  # unused (API compatibility)
        uncorrelated_sample_indices: Union[
            np.ndarray, list[int], None
        ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,  # unused (API compatibility)
        sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD,  # unused (API compatibility)
    ) -> float:
        r"""Approximate the upper confidence limit of the mean [mokashi2010]_.

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

        """
        time_series_data = np.asarray(time_series_data)

        if time_series_data.ndim != 1:
            raise CRError("time_series_data is not an array of one-dimension.")

        if confidence_coefficient <= 0.0 or confidence_coefficient >= 1.0:
            raise CRError(
                f"confidence_coefficient = {confidence_coefficient} is not "
                "in the range (0.0 1.0)."
            )

        # Initialize
        x_batch = batch(
            time_series_data,
            batch_size=batch_size,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )

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
        # the spread the more accurate. Please see ref [mokashi2010]_
        standard_error_of_mean = self.std / sqrt(number_batches)

        # Compute the t_distribution confidence interval. When using the
        # t-distribution to compute a confidence interval, df = n - 1.
        p_up = (1 + confidence_coefficient) / 2
        # Please see ref [mokashi2010]_
        upper = t_inv_cdf(p_up, number_batches - 1)

        self.upper_confidence_limit = upper * standard_error_of_mean
        assert isinstance(self.upper_confidence_limit, float)  # keeps mypy happy
        return float(self.upper_confidence_limit)  # ensures built-in float, not numpy scalar


def mser_m_ucl(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    scale: str = _DEFAULT_SCALE_METHOD,
    with_centering: bool = _DEFAULT_WITH_CENTERING,
    with_scaling: bool = _DEFAULT_WITH_SCALING,
    obj: Optional[MSER_m] = None,
) -> float:
    r"""Approximate the upper confidence limit of the mean."""
    mser = MSER_m() if obj is None else obj
    upper_confidence_limit = mser.ucl(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        batch_size=batch_size,
        scale=scale,
        with_centering=with_centering,
        with_scaling=with_scaling,
    )
    return upper_confidence_limit


def mser_m_ci(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    scale: str = _DEFAULT_SCALE_METHOD,
    with_centering: bool = _DEFAULT_WITH_CENTERING,
    with_scaling: bool = _DEFAULT_WITH_SCALING,
    obj: Optional[MSER_m] = None,
) -> tuple[float, float]:
    r"""Approximate the confidence interval of the mean [mokashi2010]_.

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
        with_scaling=with_scaling,
    )
    return confidence_limits


def mser_m_relative_half_width_estimate(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    scale: str = _DEFAULT_SCALE_METHOD,
    with_centering: bool = _DEFAULT_WITH_CENTERING,
    with_scaling: bool = _DEFAULT_WITH_SCALING,
    obj: Optional[MSER_m] = None,
) -> float:
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
        relative_half_width_estimate = mser.relative_half_width_estimate(
            time_series_data=time_series_data,
            confidence_coefficient=confidence_coefficient,
            batch_size=batch_size,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )
    except CRError as e:
        raise CRError("Failed to get the relative_half_width_estimate.") from e
    return relative_half_width_estimate
