"""MSER-m_y UCL module."""

from math import ceil, sqrt
import numpy as np
from typing import Optional, Union

from .mser_m import MSER_m
from convergence._default import \
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
    _DEFAULT_SAMPLE_METHOD
from convergence import \
    batch, \
    CVGError, \
    CVGSampleSizeError, \
    randomness_test, \
    t_inv_cdf


__all__ = [
    'MSER_m_y',
    'mser_m_y_ucl',
    'mser_m_y_ci',
    'mser_m_y_relative_half_width_estimate',
]


class MSER_m_y(MSER_m):
    r"""MSER_m_y algorithm.

    MSER_m_y [21]_ computes k batch means of size m to evaluate the MSER-m
    statistic as described in [4]_ and detect the trucation point. If the
    truncation is detected, the point estimator of the mean is the sample mean
    of all observations in the truncated data set.
    To compute the UCL, the MSER_m_y applies the von Neumann randomness test
    [14]_, [15]_ to the truncated data to find a new batch size :math:`m^*` for
    which the new batch means are approximately independent. It checks the
    randomness test on successively larger batch sizes until the test is
    finally passed and the batch means are finally determined to be
    approximately independent of each other. It starts by setting the initial
    batch size m as 1, and calculate the number of batches k′ accordingly.

    Attributes:
        significance_level (float): Significance level. A probability threshold
            below which the null hypothesis will be rejected.

    References:
        .. [21] Yousefi, S., (2011) "MSER-5Y: An Improved Version of MSER-5
                with Automatic Confidence Interval Estimation," MSc thesis,
                http://www.lib.ncsu.edu/resolver/1840.16/6923

    """

    def __init__(self):
        MSER_m.__init__(self)

        self.name = 'mser_m_y'

        # randomness test significance level \alpha = 0.2
        self.significance_level = 0.2

    def ucl(self,
            time_series_data: Union[list[float], np.ndarray],
            *,
            confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
            batch_size: int = _DEFAULT_BATCH_SIZE,
            scale: str = _DEFAULT_SCALE_METHOD,
            with_centering: bool = _DEFAULT_WITH_CENTERING,
            with_scaling: bool = _DEFAULT_WITH_SCALING,
            # unused input parmeters in
            # MSER_m ucl interface
            equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
            heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
            fft: bool = _DEFAULT_FFT,
            test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
            train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
            population_standard_deviation: Optional[float] = _DEFAULT_POPULATION_STANDARD_DEVIATION,
            si: Union[str, float, int, None] = _DEFAULT_SI,
            minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
            uncorrelated_sample_indices: Union[list[int], np.ndarray,
                                               None] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
            sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD) -> float:
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

        """
        time_series_data = np.array(time_series_data, copy=False)

        if time_series_data.ndim != 1:
            msg = 'time_series_data is not an array of one-dimension.'
            raise CVGError(msg)

        time_series_data_size = time_series_data.size

        if time_series_data_size < 10:
            msg = '{} input data points are not '.format(time_series_data_size)
            msg += 'sufficient to be used by "MSER_m_y".\n"MSER_m_y" at '
            msg += 'least needs 10 data points.'
            raise CVGSampleSizeError(msg)

        if confidence_coefficient <= 0.0 or confidence_coefficient >= 1.0:
            msg = 'confidence_coefficient = {} '.format(confidence_coefficient)
            msg += 'is not in the range (0.0 1.0).'
            raise CVGError(msg)

        batch_size = 1
        number_batches = time_series_data_size

        # Apply the randomness test of von Neumann
        # to the current set of batch means
        x_batch = np.array(time_series_data, copy=False)
        random = randomness_test(x_batch, self.significance_level)

        dependent_data = not random
        sufficient_data = True

        while dependent_data and sufficient_data:
            batch_size = ceil(1.2 * batch_size)
            number_batches = time_series_data_size // batch_size
            processed_sample_size = number_batches * batch_size

            if processed_sample_size <= time_series_data_size:
                # Batch the data
                x_batch = batch(time_series_data[:processed_sample_size],
                                batch_size=batch_size,
                                scale=scale,
                                with_centering=with_centering,
                                with_scaling=with_scaling)

                random = randomness_test(x_batch, self.significance_level)

                dependent_data = not random

            else:
                sufficient_data = False

        if dependent_data:
            batch_size = 10
            number_batches = time_series_data_size // batch_size
            processed_sample_size = number_batches * batch_size

            # Batch the data
            x_batch = batch(time_series_data[:processed_sample_size],
                            batch_size=batch_size,
                            scale=scale,
                            with_centering=with_centering,
                            with_scaling=with_scaling)

        number_batches = x_batch.size

        # Compute and set the sample mean of all observations
        # in the truncated data set
        self.mean = time_series_data.mean()

        # Compute the sample standard deviation (sample variance of the
        # truncated batch means)
        self.std = x_batch.std()
        self.sample_size = number_batches

        # Compute the standard deviation of the mean within the dataset. The
        # standard_error_of_mean provides a measurement for spread. The smaller
        # the spread the more accurate.
        standard_error_of_mean = self.std / sqrt(number_batches)

        # Compute the t_distribution confidence interval. When using the
        # t-distribution to compute a confidence interval, df = n - 1.
        p_up = (1 + confidence_coefficient) / 2
        upper = t_inv_cdf(p_up, number_batches - 1)

        self.upper_confidence_limit = upper * standard_error_of_mean
        return self.upper_confidence_limit


def mser_m_y_ucl(time_series_data: Union[list[float], np.ndarray],
                 *,
                 confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
                 batch_size: int = _DEFAULT_BATCH_SIZE,
                 scale: str = _DEFAULT_SCALE_METHOD,
                 with_centering: bool = _DEFAULT_WITH_CENTERING,
                 with_scaling: bool = _DEFAULT_WITH_SCALING,
                 obj: Optional[MSER_m_y] = None) -> float:
    """Approximate the upper confidence limit of the mean."""
    mser = MSER_m_y() if obj is None else obj
    upper_confidence_limit = mser.ucl(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        batch_size=batch_size,
        scale=scale,
        with_centering=with_centering,
        with_scaling=with_scaling)
    return upper_confidence_limit


def mser_m_y_ci(time_series_data: Union[list[float], np.ndarray],
                *,
                confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
                batch_size: int = _DEFAULT_BATCH_SIZE,
                scale: str = _DEFAULT_SCALE_METHOD,
                with_centering: bool = _DEFAULT_WITH_CENTERING,
                with_scaling: bool = _DEFAULT_WITH_SCALING,
                obj: Optional[MSER_m_y] = None) -> tuple((float, float)):
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
        obj (MSER_m_y, optional): instance of ``MSER_m_y`` (default: None)

    Returns:
        float, float: confidence interval

    """
    mser = MSER_m_y() if obj is None else obj
    confidence_limits = mser.ci(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        batch_size=batch_size,
        scale=scale,
        with_centering=with_centering,
        with_scaling=with_scaling)
    return confidence_limits


def mser_m_y_relative_half_width_estimate(
        time_series_data: Union[list[float], np.ndarray],
        *,
        confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        scale: str = _DEFAULT_SCALE_METHOD,
        with_centering: bool = _DEFAULT_WITH_CENTERING,
        with_scaling: bool = _DEFAULT_WITH_SCALING,
        obj: Optional[MSER_m_y] = None) -> float:
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
        obj (MSER_m_y, optional): instance of ``MSER_m_y`` (default: None)

    Returns:
        float: the relative half width estimate.

    """
    mser = MSER_m_y() if obj is None else obj
    try:
        relative_half_width_estimate = mser.relative_half_width_estimate(
            time_series_data=time_series_data,
            confidence_coefficient=confidence_coefficient,
            batch_size=batch_size,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling)
    except CVGError:
        raise CVGError('Failed to get the relative_half_width_estimate.')
    return relative_half_width_estimate
