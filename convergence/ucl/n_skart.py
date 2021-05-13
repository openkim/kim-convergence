"""N-SKART UCL method."""

from math import ceil, floor, fabs, sqrt
import numpy as np

from .ucl_base import UCLBase
from convergence import \
    batch, \
    CVGError, \
    cvg_warning, \
    skew, \
    randomness_test, \
    auto_correlate, \
    t_inv_cdf

__all__ = [
    'N_SKART',
    'n_skart_ucl',
    'n_skart_ci',
    'n_skart_relative_half_width_estimate',
]


class N_SKART(UCLBase):
    r"""N-Skart class.

    N-Skart is a nonsequential procedure designed to compute a half the width
    of the `confidence_coefficient%` probability interval (CI) (confidence
    interval, or credible interval) around the time-series mean.

    Notes:
        N-Skart is a variant of the method of batch means.

        N-Skart makes some modifications to the confidence interval (CI).
        These modifications account for the skewness (non-normality), and
        autocorrelation of the batch means which affect the distribution of the
        underlying Student’s t-statistic.

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
            of von Neumann [14]_ [15]_.

    References:
        .. [19] Tafazzoli, A. and Steiger, N.M. and  Wilson, J.R., (2011)
                "N-Skart: A Nonsequential Skewness- and Autoregression-Adjusted
                Batch-Means Procedure for Simulation Analysis," IEEE
                Transactions on Automatic Control, 56(2), 254-264.

    """

    def __init__(self):
        """Initialize the N_SKART class.

        Initialize a N_SKART object and set the constants.

        """
        self._reset()

        UCLBase.__init__(self)

    def _reset(self):
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

    def estimate_equilibration_length(self, time_series_data):
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
        time_series_data = np.array(time_series_data, copy=False)
        if time_series_data.ndim != 1:
            msg = 'time_series_data is not an array of one-dimension.'
            raise CVGError(msg)

        time_series_data_size = time_series_data.size

        # Minimum number of data points
        if time_series_data_size < self.k_number_batches:
            msg = '{} input data points are not '.format(time_series_data_size)
            msg += 'sufficient to be used by "N-Skart".\n"N-Skart" at '
            msg += 'least needs {} data points.'.format(self.k_number_batches)
            raise CVGError(msg)

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
        x_batch = batch(time_series_data[:processed_sample_size],
                        batch_size=self.batch_size,
                        with_centering=False,
                        with_scaling=False)

        dependent_data = True
        sufficient_data = True

        while dependent_data and sufficient_data:
            # Perform step 2 of the N-Skart algorithm

            # Compute the sample skewness of the last 80% of the current batch
            idx = self.k_number_batches // 5

            last_80_percent_x_batch = x_batch[idx:]
            last_80_percent_x_batch_skewness = skew(last_80_percent_x_batch,
                                                    bias=False)

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
            while dependent_data and \
                    self.number_batches_per_spacer < \
                    self.maximum_number_batches_per_spacer:

                # d <- d + 1
                self.number_batches_per_spacer += 1

                idx = self.number_batches_per_spacer

                spaced_x_batch = x_batch[idx::idx + 1]

                # k'
                self.kp_number_batches = spaced_x_batch.size

                random = randomness_test(spaced_x_batch,
                                         self.significance_level)

                dependent_data = not random

            # Perform step 4 of the N-Skart algorithm
            if dependent_data:
                # step 4
                # m <- ceil(sqrt(2) * m)
                batch_size = ceil(sqrt(2.) * self.batch_size)
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
                    x_batch = batch(time_series_data[:processed_sample_size],
                                    batch_size=self.batch_size)
                else:
                    msg = '{} number of input '.format(time_series_data_size)
                    msg += 'data points is not sufficient to be used by '
                    msg += '"N-Skart" method.\n"N-Skart" at least needs '
                    msg += '{} = '.format(processed_sample_size)
                    msg += '{} x {}'.format(k_number_batches, batch_size)
                    msg += ' data points.\n'

                    sufficient_data = False

        if sufficient_data:
            # Perform step 5 of the N-Skart algorithm
            # w <- d * m
            truncate_index = self.number_batches_per_spacer * self.batch_size
            return True, truncate_index

        cvg_warning(msg)
        return False, time_series_data_size - 1

    def ucl(self,
            time_series_data,
            *,
            confidence_coefficient=0.95,
            equilibration_length_estimate=0,
            fft=True,
            # unused input parmeters in
            # N_SKART ucl interface
            heidel_welch_number_points=None,
            batch_size=None,
            scale=None,
            with_centering=None,
            with_scaling=None,
            test_size=None,
            train_size=None,
            population_standard_deviation=None,
            si=None,
            minimum_correlation_time=None,
            uncorrelated_sample_indices=None,
            sample_method=None):
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
        time_series_data = np.array(time_series_data, copy=False)

        if time_series_data.ndim != 1:
            msg = 'time_series_data is not an array of one-dimension.'
            raise CVGError(msg)

        if not isinstance(equilibration_length_estimate, int):
            msg = 'equilibration_length_estimate must be an `int`.'
            raise CVGError(msg)

        if confidence_coefficient <= 0.0 or confidence_coefficient >= 1.0:
            msg = 'confidence_coefficient = {} '.format(confidence_coefficient)
            msg += 'is not in the range (0.0 1.0).'
            raise CVGError(msg)

        # Perform step 5 of the N-Skart algorithm

        # step 5a

        time_series_data_size = time_series_data.size

        if self.kp_number_batches != self.k_number_batches:
            # Reinflate the batch count, k' <- min(ceil(k'(1 / 0.9)^b), k)
            kp = ceil(self.kp_number_batches *
                      (1.0 / 0.9) ** self.randomness_test_counter)

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
            msg = '{} input data points are '.format(time_series_data_size)
            msg += 'not sufficient to be used by "N-Skart".\n"N-Skart" at '
            msg += 'least needs {} data points.'.format(processed_sample_size)
            raise CVGError(msg)

        # step 5b

        # update the length of the warm-up period
        idx = time_series_data_size - processed_sample_size

        # w <- w + (N' - k' m)
        equilibration_length_estimate += idx

        sliced_time = time_series_data[idx:]

        # Batch the data
        x_batch = batch(sliced_time, batch_size=self.batch_size)

        # Perform step 6 of the N-Skart algorithm

        # compute and set the mean to be used later in interval method
        self.mean = x_batch.mean()
        self.std = x_batch.std()

        # compute the sample variance
        x_batch_var = x_batch.var(ddof=1)

        # compute the sample estimator of the lag-one correlation
        lag1_correlation = \
            auto_correlate(x_batch,
                           nlags=1,
                           fft=(self.kp_number_batches > 30 and fft))[1]

        # compute the correlation adjustment A <- (1 + \phi) / (1 - \phi)
        correlation_adjustment = \
            (1 + lag1_correlation) / (1 - lag1_correlation)

        # Perform step 7 of the N-Skart algorithm

        # number of batches per spacer, d' <- ceil(w / m)
        batches_per_spacer = \
            ceil(equilibration_length_estimate / self.batch_size)

        # modification to the original paper
        spaced_x_batch = x_batch[batches_per_spacer::batches_per_spacer+1]

        # number of spaced batches, k'' = 1 + floor((k' - 1) / (d' + 1))
        spaced_x_batch_size = 1 + \
            (self.kp_number_batches - 1) // (batches_per_spacer + 1)
        # spaced_x_batch_size = spaced_x_batch.size

        # compute skewness
        spaced_x_batch_skewness = skew(spaced_x_batch, bias=False)

        # compute beta <- skew / (6 sqrt(k''))
        beta = spaced_x_batch_skewness / (6 * sqrt(spaced_x_batch_size))

        # Compute the t_distribution confidence interval. When using the
        # t-distribution to compute a confidence interval, df = n - 1.
        upper_p = (1.0 + confidence_coefficient) / 2
        upper = t_inv_cdf(upper_p, spaced_x_batch_size - 1)

        skewness_adjustment = (
            (1 + 6 * beta * (upper - beta))**(1 / 3) - 1) / (2 * beta)

        upper_confidence_limit = skewness_adjustment * \
            sqrt(correlation_adjustment * x_batch_var / self.kp_number_batches)
        return upper_confidence_limit


def n_skart_ucl(time_series_data,
                *,
                confidence_coefficient=0.95,
                equilibration_length_estimate=0,
                fft=True,
                obj=None):
    """Approximate the upper confidence limit of the mean."""
    n_skart = N_SKART() if obj is None else obj
    upper_confidence_limit = n_skart.ucl(
        time_series_data=time_series_data,
        equilibration_length_estimate=equilibration_length_estimate,
        confidence_coefficient=confidence_coefficient,
        fft=fft)
    return upper_confidence_limit


def n_skart_ci(time_series_data,
               *,
               confidence_coefficient=0.95,
               equilibration_length_estimate=0,
               fft=True,
               obj=None):
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
        fft=fft)
    return confidence_limits


def n_skart_relative_half_width_estimate(
        time_series_data,
        *,
        confidence_coefficient=0.95,
        equilibration_length_estimate=0,
        fft=True,
        obj=None):
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
    relative_half_width_estimate = n_skart.relative_half_width_estimate(
        time_series_data=time_series_data,
        equilibration_length_estimate=equilibration_length_estimate,
        confidence_coefficient=confidence_coefficient,
        fft=fft)
    return relative_half_width_estimate
