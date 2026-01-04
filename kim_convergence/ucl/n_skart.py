r"""N-SKART UCL module."""

from math import ceil, floor, fabs, sqrt
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
    _DEFAULT_NSKIP,
    _DEFAULT_IGNORE_END,
    _DEFAULT_NUMBER_OF_CORES,
)
from .ucl_base import UCLBase
from kim_convergence import (
    batch,
    CRError,
    CRSampleSizeError,
    cr_warning,
    skew,
    randomness_test,
    auto_correlate,
    t_inv_cdf,
)


__all__ = [
    "N_SKART",
    "n_skart_ci",
    "n_skart_relative_half_width_estimate",
    "n_skart_ucl",
]


class N_SKART(UCLBase):
    r"""N-Skart algorithm.

    N-Skart [tafazzoli2011]_ is a nonsequential procedure designed to compute a
    half the width of the `confidence_coefficient%` probability interval (CI)
    (confidence interval, or credible interval) around the time-series mean.

    Note:
        N-Skart is a variant of the method of batch means.

        N-Skart makes some modifications to the confidence interval (CI).
        These modifications account for the skewness (non-normality), and
        autocorrelation of the batch means which affect the distribution of the
        underlying Student's t-statistic.

    Attributes:
        k_number_batches (int): number of nonspaced (adjacent) batches of size
            ``batch_size``.
        kp_number_batches (int): number of nonspaced (adjacent) batches.
        batch_size (int): bacth size.
        number_batches_per_spacer (int): number of batches per spacer.
        maximum_number_batches_per_spacer (int): maximum number of batches per
            spacer.
        significance_level (float): Significance level. A probability threshold
            below which the null hypothesis will be rejected.
        randomness_test_counter (int): counter for applying the randomness test
            of von Neumann [vonneumann1941]_ [vonneumann1941b]_.

    """

    def __init__(self):
        r"""Initialize the N_SKART class.

        Initialize a N_SKART object and set the constants.

        """
        UCLBase.__init__(self)

        self.name = "n_skart"

        self._reset()

    def _reset(self) -> None:
        """reset the parmaters."""
        # k <- 1280
        self.k_number_batches = 1280
        # k' <- 1280
        self.kp_number_batches = 1280
        # m <- 1
        self.batch_size = 1
        # d <- 0
        self.number_batches_per_spacer = 0
        # d* <- 10
        self.maximum_number_batches_per_spacer = 10
        # randomness test significance level \alpha = 0.2
        self.significance_level = 0.2
        # Number of times the batch count has been deflated
        # in the randomness test b <- 0
        self.randomness_test_counter = 0

        # UCLBase reset method
        self.reset()

    def estimate_equilibration_length(
        self,
        time_series_data: Union[np.ndarray, list[float]],
        *,
        si: Union[str, float, int, None] = _DEFAULT_SI,
        nskip: Optional[int] = _DEFAULT_NSKIP,  # unused (API compatibility)
        fft: bool = _DEFAULT_FFT,
        minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
        ignore_end: Union[int, float, None] = _DEFAULT_IGNORE_END,  # unused (API compatibility)
        number_of_cores: int = _DEFAULT_NUMBER_OF_CORES,  # unused (API compatibility)
        batch_size: int = _DEFAULT_BATCH_SIZE,
        scale: str = _DEFAULT_SCALE_METHOD,
        with_centering: bool = _DEFAULT_WITH_CENTERING,
        with_scaling: bool = _DEFAULT_WITH_SCALING,
    ) -> tuple[bool, int]:
        r"""Estimate the equilibration point in a time series data.

        Estimate the equilibration point in a time series data using the
        N-Skart algorithm.

        Args:
            time_series_data (array_like, 1d): time series data.

        Returns:
            bool, int: truncated, equilibration index.
                Truncation point is the index to truncate.

        Note:
            if N-Skart does not detect the equilibration it will return
            truncated as False and the equilibration index equals to the last
            index in the time series data.

        """
        time_series_data = np.asarray(time_series_data)
        if time_series_data.ndim != 1:
            raise CRError("time_series_data is not an array of one-dimension.")

        time_series_data_size = time_series_data.size

        # Minimum number of data points
        if time_series_data_size < self.k_number_batches:
            raise CRSampleSizeError(
                f"{time_series_data_size} input data points are not "
                'sufficient to be used by "N-Skart".\n"N-Skart" at '
                f"least needs {self.k_number_batches} data points."
            )

        # Reset the parameters for run-length control
        # cases, where we call this function in a loop
        self._reset()

        # Compute the sample skewness of the last 80% of the current data

        idx = time_series_data_size // 5

        # slice a numpy array, the memory is shared between the slice and
        # the original
        last_80_percent_data = time_series_data[idx:]
        last_80_percent_data_skewness = skew(last_80_percent_data, bias=False)

        # Set the initial batch size m
        if fabs(last_80_percent_data_skewness) > 4.0:
            # m <- min(16, floor(N / 1280))
            batch_size = time_series_data_size // self.k_number_batches
            self.batch_size = min(16, batch_size)

        # N-Skart uses the initial n = 1280 x m observations
        # of the overall sample of size N
        processed_sample_size = self.k_number_batches * self.batch_size

        # Batch the data, Y_j(m) : j = 1 ... k
        x_batch = batch(
            time_series_data[:processed_sample_size],
            batch_size=self.batch_size,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )

        dependent_data = True
        sufficient_data = True

        while dependent_data and sufficient_data:
            # Perform step 2 of the N-Skart algorithm

            # Compute the sample skewness of the last 80% of the current batch
            idx = self.k_number_batches // 5

            last_80_percent_x_batch = x_batch[idx:]
            last_80_percent_x_batch_skewness = skew(last_80_percent_x_batch, bias=False)

            if fabs(last_80_percent_x_batch_skewness) > 0.5:
                # reset the maximum number of batches, d* <- 3
                self.maximum_number_batches_per_spacer = 3

            # Perform step 3 of the N-Skart algorithm

            # Apply the randomness test of von Neumann
            # to the current set of batch means
            random = randomness_test(x_batch, self.significance_level)

            # step 3a
            if random:
                # k' <- k
                self.kp_number_batches = self.k_number_batches
                dependent_data = False

            # step 3b - 3d
            while (
                dependent_data
                and self.number_batches_per_spacer
                < self.maximum_number_batches_per_spacer
            ):

                # d <- d + 1
                self.number_batches_per_spacer += 1

                idx = self.number_batches_per_spacer

                spaced_x_batch = x_batch[idx::idx + 1]

                # k'
                self.kp_number_batches = spaced_x_batch.size

                random = randomness_test(spaced_x_batch, self.significance_level)

                dependent_data = not random

            # Perform step 4 of the N-Skart algorithm
            if dependent_data:
                # step 4
                # m <- ceil(sqrt(2) * m)
                batch_size = ceil(sqrt(2.0) * self.batch_size)
                # k <- ceil(0.9 * k)
                k_number_batches = ceil(0.9 * self.k_number_batches)
                # n <- k * m
                processed_sample_size = k_number_batches * batch_size

                if processed_sample_size <= time_series_data_size:
                    # m <- ceil(sqrt(2) * m)
                    self.batch_size = batch_size
                    # k <- ceil(0.9 * k)
                    self.k_number_batches = k_number_batches
                    # d <- 0
                    self.number_batches_per_spacer = 0
                    # d* <- 10
                    self.maximum_number_batches_per_spacer = 10
                    # b <- b + 1
                    self.randomness_test_counter += 1

                    # Rebatch the data
                    x_batch = batch(
                        time_series_data[:processed_sample_size],
                        batch_size=self.batch_size,
                        scale=scale,
                        with_centering=with_centering,
                        with_scaling=with_scaling,
                    )
                else:
                    cr_warning(
                        f"{time_series_data_size} number of input data points "
                        'is not sufficient to be used by "N-Skart" method.\n'
                        f'"N-Skart" at least needs {processed_sample_size} = '
                        f"{k_number_batches} x {batch_size} data points.\n"
                    )

                    sufficient_data = False

        if sufficient_data:
            # Perform step 5 of the N-Skart algorithm
            # w <- d * m
            truncate_index = self.number_batches_per_spacer * self.batch_size

            self.set_si(
                time_series_data=time_series_data[truncate_index:],
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
            )

            return True, truncate_index

        self.si = None
        return False, time_series_data_size

    def _ucl_impl(
        self,
        time_series_data: Union[np.ndarray, list[float]],
        *,
        confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
        equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
        fft: bool = _DEFAULT_FFT,
        heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,  # unused (API compatibility)
        batch_size: int = _DEFAULT_BATCH_SIZE,  # unused (API compatibility)
        scale: str = _DEFAULT_SCALE_METHOD,
        with_centering: bool = _DEFAULT_WITH_CENTERING,
        with_scaling: bool = _DEFAULT_WITH_SCALING,
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
        r"""Approximate the upper confidence limit of the mean.

        Args:
            time_series_data (array_like, 1d): time series data.
            equilibration_length_estimate (int, optional): an estimate for the
                equilibration length.
            confidence_coefficient (float, optional): probability (or confidence
                interval) and must be between 0.0 and 1.0, and represents the
                confidence for calculation of relative halfwidths estimation.
                (default: 0.95)
            fft (bool, optional): if ``True``, use FFT convolution. FFT should
                be preferred for long time series. (default: True)

        Returns:
            float: upper_confidence_limit
                The correlation-adjusted estimate of variance of the sample
                mean, based on the skewness-adjusted critical values of
                Student's t-ratio.

        """
        time_series_data = np.asarray(time_series_data)

        if time_series_data.ndim != 1:
            raise CRError("time_series_data is not an array of one-dimension.")

        if not isinstance(equilibration_length_estimate, int):
            raise CRError("equilibration_length_estimate must be an `int`.")

        if confidence_coefficient <= 0.0 or confidence_coefficient >= 1.0:
            raise CRError(
                f"confidence_coefficient = {confidence_coefficient} is not "
                "in the range (0.0 1.0)."
            )

        # Perform step 5 of the N-Skart algorithm

        # step 5a

        time_series_data_size = time_series_data.size

        if self.kp_number_batches != self.k_number_batches:
            # Reinflate the batch count, k' <- min(ceil(k'(1 / 0.9)^b), k)
            kp = ceil(
                self.kp_number_batches * (1.0 / 0.9) ** self.randomness_test_counter
            )

            self.kp_number_batches = min(kp, self.k_number_batches)

        # compute additional inflation factor

        # (k' m)
        processed_sample_size = self.kp_number_batches * self.batch_size

        # compute the inflation factor, f <- sqrt(N' / (k' m))
        inflation_factor = sqrt(time_series_data_size / processed_sample_size)

        # reset the truncated batch count k' <- min(floor(f k'), 1024)
        kp = floor(inflation_factor * self.kp_number_batches)

        if kp < 1024:
            # k' <- floor(f k')
            self.kp_number_batches = kp
            # reset the bacth size, m <- floor(f m)
            self.batch_size = floor(inflation_factor * self.batch_size)
        else:
            # k' <- 1024
            self.kp_number_batches = 1024
            # set the bacth size, m <- floor(N' / 1024)
            self.batch_size = time_series_data_size // 1024

        # (k' m)
        processed_sample_size = self.kp_number_batches * self.batch_size

        # Minimum number of data points
        if time_series_data_size < processed_sample_size:
            msg = (
                f"{time_series_data_size} input data points are not "
                'sufficient to be used by "N-Skart".\n"N-Skart" at '
                f"least needs {processed_sample_size} data points."
            )
            cr_warning(msg)
            raise CRSampleSizeError(msg)

        # step 5b

        # update the length of the warm-up period
        idx = time_series_data_size - processed_sample_size

        # w <- w + (N' - k' m)
        equilibration_length_estimate += idx

        sliced_time = time_series_data[idx:]

        # Batch the data
        x_batch = batch(
            sliced_time,
            batch_size=self.batch_size,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )

        # Perform step 6 of the N-Skart algorithm

        # compute and set the mean to be used later in interval method
        self.mean = x_batch.mean()
        self.std = x_batch.std()
        self.sample_size = x_batch.size

        # compute the sample variance
        x_batch_var = x_batch.var(ddof=1)

        # compute the sample estimator of the lag-one correlation
        lag1_correlation = auto_correlate(
            x_batch, nlags=1, fft=(fft and x_batch.size > 30)
        )[1]

        # compute the correlation adjustment A <- (1 + \phi) / (1 - \phi)
        correlation_adjustment = (1 + lag1_correlation) / (1 - lag1_correlation)

        # Perform step 7 of the N-Skart algorithm

        # number of batches per spacer, d' <- ceil(w / m)
        batches_per_spacer = ceil(equilibration_length_estimate / self.batch_size)

        # modification to the original paper
        spaced_x_batch = x_batch[batches_per_spacer::batches_per_spacer + 1]

        # number of spaced batches, k'' = 1 + floor((k' - 1) / (d' + 1))
        spaced_x_batch_size = 1 + (self.kp_number_batches - 1) // (
            batches_per_spacer + 1
        )
        # spaced_x_batch_size = spaced_x_batch.size

        # compute skewness
        spaced_x_batch_skewness = skew(spaced_x_batch, bias=False)

        # compute beta <- skew / (6 sqrt(k''))
        beta = spaced_x_batch_skewness / (6 * sqrt(spaced_x_batch_size))

        # Compute the t_distribution confidence interval. When using the
        # t-distribution to compute a confidence interval, df = n - 1.
        upper_p = (1.0 + confidence_coefficient) / 2
        upper = t_inv_cdf(upper_p, spaced_x_batch_size - 1)

        skewness_adjustment = ((1 + 6 * beta * (upper - beta)) ** (1 / 3) - 1) / (
            2 * beta
        )

        self.upper_confidence_limit = skewness_adjustment * sqrt(
            correlation_adjustment * x_batch_var / self.kp_number_batches
        )
        assert isinstance(self.upper_confidence_limit, float)  # keeps mypy happy
        return float(self.upper_confidence_limit)  # ensures built-in float, not numpy scalar


def n_skart_ucl(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
    equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
    fft: bool = _DEFAULT_FFT,
    obj: Optional[N_SKART] = None,
) -> float:
    """Approximate the upper confidence limit of the mean."""
    n_skart = N_SKART() if obj is None else obj
    upper_confidence_limit = n_skart.ucl(
        time_series_data=time_series_data,
        equilibration_length_estimate=equilibration_length_estimate,
        confidence_coefficient=confidence_coefficient,
        fft=fft,
    )
    return upper_confidence_limit


def n_skart_ci(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
    equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
    fft: bool = _DEFAULT_FFT,
    obj: Optional[N_SKART] = None,
) -> tuple[float, float]:
    r"""Approximate the confidence interval of the mean.

    Args:
        time_series_data (array_like, 1d): time series data.
        equilibration_length_estimate (int, optional): an estimate for the
            equilibration length.
        confidence_coefficient (float, optional): probability (or confidence
            interval) and must be between 0.0 and 1.0, and represents the
            confidence for calculation of relative halfwidths estimation.
            (default: 0.95)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should
            be preferred for long time series. (default: True)
        obj (N_SKART, optional): instance of ``N_SKART`` (default: None)


    Returns:
        float, float: confidence interval

    """
    n_skart = N_SKART() if obj is None else obj
    confidence_limits = n_skart.ci(
        time_series_data=time_series_data,
        equilibration_length_estimate=equilibration_length_estimate,
        confidence_coefficient=confidence_coefficient,
        fft=fft,
    )
    return confidence_limits


def n_skart_relative_half_width_estimate(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient=_DEFAULT_CONFIDENCE_COEFFICIENT,
    equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
    fft: bool = _DEFAULT_FFT,
    obj: Optional[N_SKART] = None,
) -> float:
    r"""Get the relative half width estimate.

    The relative half width estimate is the confidence interval
    half-width or upper confidence limit (UCL) divided by the sample mean.

    The UCL is calculated as a `confidence_coefficient%` confidence
    interval for the mean, using the portion of the time series data, which
    is in the stationarity region.

    Args:
        time_series_data (array_like, 1d): time series data.
        equilibration_length_estimate (int, optional): an estimate for the
            equilibration length.
        confidence_coefficient (float, optional): probability (or confidence
            interval) and must be between 0.0 and 1.0, and represents the
            confidence for calculation of relative halfwidths estimation.
            (default: 0.95)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should
            be preferred for long time series. (default: True)
        obj (N_SKART, optional): instance of ``N_SKART`` (default: None)

    Returns:
        float: the relative half width estimate.

    """
    n_skart = N_SKART() if obj is None else obj
    try:
        relative_half_width_estimate = n_skart.relative_half_width_estimate(
            time_series_data=time_series_data,
            equilibration_length_estimate=equilibration_length_estimate,
            confidence_coefficient=confidence_coefficient,
            fft=fft,
        )
    except CRError as e:
        raise CRError("Failed to get the relative_half_width_estimate.") from e
    return relative_half_width_estimate
