r"""UncorrelatedSamples UCL module."""

from math import sqrt
import numpy as np
from typing import Optional, Union

from kim_convergence._default import (
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
)
from .ucl_base import UCLBase
from kim_convergence import (
    CRError,
    CRSampleSizeError,
    cr_warning,
    t_inv_cdf,
    time_series_data_si,
    uncorrelated_time_series_data_sample_indices,
    uncorrelated_time_series_data_samples,
)

__all__ = [
    "UncorrelatedSamples",
    "uncorrelated_samples_ucl",
    "uncorrelated_samples_ci",
    "uncorrelated_samples_relative_half_width_estimate",
]

SAMPLING_METHODS = ("uncorrelated", "random", "block_averaged")


class UncorrelatedSamples(UCLBase):
    r"""UncorrelatedSamples algorithm."""

    def __init__(self):
        UCLBase.__init__(self)

        self.name = "uncorrelated_sample"

    def _ucl_impl(
        self,
        time_series_data: Union[np.ndarray, list[float]],
        *,
        confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
        population_standard_deviation: Optional[
            float
        ] = _DEFAULT_POPULATION_STANDARD_DEVIATION,
        si: Union[str, float, int, None] = _DEFAULT_SI,
        fft: bool = _DEFAULT_FFT,
        minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
        uncorrelated_sample_indices: Union[
            np.ndarray, list[int], None
        ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
        sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD,
        equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,  # unused (API compatibility)
        heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,  # unused (API compatibility)
        batch_size: int = _DEFAULT_BATCH_SIZE,  # unused (API compatibility)
        scale: str = _DEFAULT_SCALE_METHOD,  # unused (API compatibility)
        with_centering: bool = _DEFAULT_WITH_CENTERING,  # unused (API compatibility)
        with_scaling: bool = _DEFAULT_WITH_SCALING,  # unused (API compatibility)
        test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,  # unused (API compatibility)
        train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,  # unused (API compatibility)
    ) -> float:
        r"""Approximate the upper confidence limit of the mean.

        - If the population standard deviation is known, and
          `population_standard_deviation` is given,

          .. math::

                UCL = t_{\alpha,d} \left(\frac{\text population\ standard\ deviation}{\sqrt{n}}\right)

        - If the population standard deviation is unknown, the sample standard
          deviation is estimated and be used as `sample_standard_deviation`,

          .. math::

                UCL = t_{\alpha,d} \left(\frac{\text sample\ standard\ deviation}{\sqrt{n}}\right)

        In both cases, the ``Student's t`` distribution is used as the critical
        value. This value depends on the `confidence_coefficient` and the
        degrees of freedom, which is found by subtracting one from the number
        of observations.

        Args:
            time_series_data (array_like, 1d): time series data.
            confidence_coefficient (float, optional): probability (or
                confidence interval) and must be between 0.0 and 1.0, and
                represents the confidence for calculation of relative
                halfwidths estimation. (default: 0.95)
            population_standard_deviation (float, optional): population
                standard deviation. (default: None)
            si (float, or str, optional): estimated statistical inefficiency.
                (default: None)
            fft (bool, optional): if True, use FFT convolution. FFT should be
                preferred for long time series. (default: True)
            minimum_correlation_time (int, optional): minimum amount of
                correlation function to compute. The algorithm terminates after
                computing the correlation time out to minimum_correlation_time
                when the correlation function first goes negative.
                (default: None)
            uncorrelated_sample_indices (array_like, 1d, optional): indices of
                uncorrelated subsamples of the time series data.
                (default: None)
            sample_method (str, optional): sampling method, one of the
                ``uncorrelated``, ``random``, or ``block_averaged``.
                (default: None)

        Returns:
            float: upper_confidence_limit
                The approximately unbiased estimate of variance of the sample
                mean.

        """
        time_series_data = np.asarray(time_series_data)

        if time_series_data.ndim != 1:
            raise CRError("time_series_data is not an array of one-dimension.")

        time_series_data_size = time_series_data.size

        if time_series_data_size < 5:
            raise CRSampleSizeError(
                f"{time_series_data_size} input data points are not sufficient "
                'to be used by "UCL".\n"UCL" at least needs 5 data points.'
            )

        if confidence_coefficient <= 0.0 or confidence_coefficient >= 1.0:
            raise CRError(
                f"confidence_coefficient = {confidence_coefficient} is not "
                "in the range (0.0 1.0)."
            )

        self.si = time_series_data_si(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
        )

        if uncorrelated_sample_indices is None:
            self.indices = uncorrelated_time_series_data_sample_indices(
                time_series_data=time_series_data,
                si=self.si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
            )
        else:
            self.indices = np.array(uncorrelated_sample_indices, copy=True)

        uncorrelated_samples = uncorrelated_time_series_data_samples(
            time_series_data=time_series_data,
            si=self.si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            uncorrelated_sample_indices=self.indices,
            sample_method=sample_method,
        )

        # Degrees of freedom
        uncorrelated_samples_size = uncorrelated_samples.size

        if uncorrelated_samples_size < 5:
            msg = (
                f"{uncorrelated_samples_size} uncorrelated sample points are "
                'not sufficient to be used by "UCL".'
            )
            cr_warning(msg)
            if uncorrelated_samples_size < 2:
                raise CRSampleSizeError(msg)

        # compute mean
        self.mean = uncorrelated_samples.mean()

        # Compute the sample standard deviation
        self.std = uncorrelated_samples.std()
        self.sample_size = uncorrelated_samples_size

        # If the population standard deviation is unknown
        if population_standard_deviation is None:
            # Compute the standard deviation of the mean within the dataset.
            # The standard_error_of_mean provides a measurement for spread.
            # The smaller the spread the more accurate.
            standard_error_of_mean = self.std / sqrt(uncorrelated_samples_size)

        # If the population standard deviation is known
        else:
            standard_error_of_mean = population_standard_deviation / sqrt(
                uncorrelated_samples_size
            )

        # Compute the t_distribution confidence interval. When using the
        # t-distribution to compute a confidence interval, df = n - 1.
        p_up = (1 + confidence_coefficient) / 2
        upper = t_inv_cdf(p_up, uncorrelated_samples_size - 1)

        self.upper_confidence_limit = upper * standard_error_of_mean
        assert isinstance(self.upper_confidence_limit, float)  # keeps mypy happy
        return float(self.upper_confidence_limit)  # ensures built-in float, not numpy scalar


def uncorrelated_samples_ucl(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
    population_standard_deviation: Optional[
        float
    ] = _DEFAULT_POPULATION_STANDARD_DEVIATION,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
    uncorrelated_sample_indices: Union[
        np.ndarray, list[int], None
    ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
    sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD,
    obj: Optional[UncorrelatedSamples] = None,
) -> float:
    r"""Approximate the upper confidence limit of the mean."""
    uncorrelated_samples = UncorrelatedSamples() if obj is None else obj
    upper_confidence_limit = uncorrelated_samples.ucl(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        population_standard_deviation=population_standard_deviation,
        si=si,
        fft=fft,
        minimum_correlation_time=minimum_correlation_time,
        uncorrelated_sample_indices=uncorrelated_sample_indices,
        sample_method=sample_method,
    )
    return upper_confidence_limit


def uncorrelated_samples_ci(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
    population_standard_deviation: Optional[
        float
    ] = _DEFAULT_POPULATION_STANDARD_DEVIATION,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
    uncorrelated_sample_indices: Union[
        np.ndarray, list[int], None
    ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
    sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD,
    obj: Optional[UncorrelatedSamples] = None,
) -> tuple[float, float]:
    r"""Approximate the confidence interval of the mean.

    - If the population standard deviation is known, and
      `population_standard_deviation` is given,

      .. math::

            UCL = t_{\alpha,d} \left(\frac{\text population\ standard\ deviation}{\sqrt{n}}\right)

    - If the population standard deviation is unknown, the sample standard
      deviation is estimated and be used as `sample_standard_deviation`,

      .. math::

            UCL = t_{\alpha,d} \left(\frac{\text sample\ standard\ deviation}{\sqrt{n}}\right)

    In both cases, the ``Student's t`` distribution is used as the critical
    value. This value depends on the `confidence_coefficient` and the
    degrees of freedom, which is found by subtracting one from the number
    of observations.

    Confidence limits for the mean are interval estimates. Interval
    estimates are often desirable because instead of a single estimate for
    the mean, a confidence interval generates a lower and upper limit. It
    indicates how much uncertainty there is in our estimation of the true
    mean. The narrower the gap, the more precise our estimate is.

    Confidence limits are defined as :math:`\bar{Y} \pm UCL,` where
    :math:`\bar{Y}` is the sample mean, and :math:`UCL` is the approximate
    upper confidence limit of the mean.

    Args:
        time_series_data (array_like, 1d): time series data.
        confidence_coefficient (float, optional): probability (or
            confidence interval) and must be between 0.0 and 1.0, and
            represents the confidence for calculation of relative
            halfwidths estimation. (default: 0.95)
        population_standard_deviation (float, optional): population
            standard deviation. (default: None)
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool, optional): if True, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of
            correlation function to compute. The algorithm terminates after
            computing the correlation time out to minimum_correlation_time
            when the correlation function first goes negative.
            (default: None)
        uncorrelated_sample_indices (array_like, 1d, optional): indices of
            uncorrelated subsamples of the time series data.
            (default: None)
        sample_method (str, optional): sampling method, one of the
            ``uncorrelated``, ``random``, or ``block_averaged``.
            (default: None)
        obj (UncorrelatedSamples, optional): instance of
            ``UncorrelatedSamples`` (default: None)

    Returns:
        float, float: confidence interval
            The approximately unbiased estimate of confidence Limits
            for the mean.

    """
    uncorrelated_samples = UncorrelatedSamples() if obj is None else obj
    confidence_limits = uncorrelated_samples.ci(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        population_standard_deviation=population_standard_deviation,
        si=si,
        fft=fft,
        minimum_correlation_time=minimum_correlation_time,
        uncorrelated_sample_indices=uncorrelated_sample_indices,
        sample_method=sample_method,
    )
    return confidence_limits


def uncorrelated_samples_relative_half_width_estimate(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
    population_standard_deviation: Optional[
        float
    ] = _DEFAULT_POPULATION_STANDARD_DEVIATION,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
    uncorrelated_sample_indices: Union[
        np.ndarray, list[int], None
    ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
    sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD,
    obj: Optional[UncorrelatedSamples] = None,
) -> float:
    r"""Get the relative half width estimate.

    The relative half width estimate is the confidence interval
    half-width or upper confidence limit (UCL) divided by the sample mean.

    The UCL is calculated as a `confidence_coefficient%` confidence
    interval for the mean, using the portion of the time series data, which
    is in the stationarity region.

    Args:
        time_series_data (array_like, 1d): time series data.
        confidence_coefficient (float, optional): probability (or
            confidence interval) and must be between 0.0 and 1.0, and
            represents the confidence for calculation of relative
            halfwidths estimation. (default: 0.95)
        population_standard_deviation (float, optional): population
            standard deviation. (default: None)
        si (float, or str, optional): estimated statistical inefficiency.
            (default: None)
        fft (bool, optional): if True, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of
            correlation function to compute. The algorithm terminates after
            computing the correlation time out to minimum_correlation_time
            when the correlation function first goes negative.
            (default: None)
        uncorrelated_sample_indices (array_like, 1d, optional): indices of
            uncorrelated subsamples of the time series data.
            (default: None)
        sample_method (str, optional): sampling method, one of the
            ``uncorrelated``, ``random``, or ``block_averaged``.
            (default: None)
        obj (UncorrelatedSamples, optional): instance of
            ``UncorrelatedSamples`` (default: None)

    Returns:
        float: the relative half width estimate

    """
    uncorrelated_samples = UncorrelatedSamples() if obj is None else obj
    try:
        relative_half_width_estimate = (
            uncorrelated_samples.relative_half_width_estimate(
                time_series_data=time_series_data,
                confidence_coefficient=confidence_coefficient,
                population_standard_deviation=population_standard_deviation,
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
                uncorrelated_sample_indices=uncorrelated_sample_indices,
                sample_method=sample_method,
            )
        )
    except CRError as e:
        raise CRError("Failed to get the relative_half_width_estimate.") from e
    return relative_half_width_estimate
