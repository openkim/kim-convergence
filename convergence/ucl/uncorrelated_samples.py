"""UncorrelatedSamples UCL method."""

from math import sqrt
import numpy as np

from .ucl_base import UCLBase
from convergence import \
    CVGError, \
    cvg_warning, \
    t_inv_cdf, \
    time_series_data_si, \
    uncorrelated_time_series_data_sample_indices, \
    uncorrelated_time_series_data_samples

__all__ = [
    'UncorrelatedSamples',
    'uncorrelated_samples_ucl',
    'uncorrelated_samples_ci',
    'uncorrelated_samples_relative_half_width_estimate',
]

SAMPLING_METHODS = ('uncorrelated', 'random', 'block_averaged')


class UncorrelatedSamples(UCLBase):
    """UncorrelatedSamples class."""

    def __init__(self):
        UCLBase.__init__(self)

    def ucl(self,
            time_series_data,
            *,
            confidence_coefficient=0.95,
            population_standard_deviation=None,
            si=None,
            fft=True,
            minimum_correlation_time=None,
            uncorrelated_sample_indices=None,
            sample_method=None,
            # unused input parmeters in
            # UncorrelatedSamples ucl interface
            equilibration_length_estimate=None,
            heidel_welch_number_points=None,
            batch_size=None,
            scale=None,
            with_centering=None,
            with_scaling=None,
            test_size=None,
            train_size=None):
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
        time_series_data = np.array(time_series_data, copy=False)

        if time_series_data.ndim != 1:
            msg = 'time_series_data is not an array of one-dimension.'
            raise CVGError(msg)

        time_series_data_size = time_series_data.size

        if time_series_data_size < 5:
            msg = '{} input data points '.format(time_series_data_size)
            msg += 'are not sufficient to be used by "UCL".\n'
            msg += '"UCL" at least needs 5 data points.'
            raise CVGError(msg)

        if confidence_coefficient <= 0.0 or confidence_coefficient >= 1.0:
            msg = 'confidence_coefficient = {} '.format(confidence_coefficient)
            msg += 'is not in the range (0.0 1.0).'
            raise CVGError(msg)

        self.si_ = time_series_data_si(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time)

        if uncorrelated_sample_indices is None:
            self.indices_ = uncorrelated_time_series_data_sample_indices(
                time_series_data=time_series_data,
                si=self.si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time)
        else:
            self.indices_ = np.array(uncorrelated_sample_indices, copy=True)

        uncorrelated_samples = uncorrelated_time_series_data_samples(
            time_series_data=time_series_data,
            si=self.si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            uncorrelated_sample_indices=self.indices,
            sample_method=sample_method)

        # Degrees of freedom
        uncorrelated_samples_size = uncorrelated_samples.size

        if uncorrelated_samples_size < 5:
            if uncorrelated_samples_size < 2:
                msg = '{} uncorrelated '.format(uncorrelated_samples_size)
                msg += 'sample points are not sufficient to be used by "UCL".'
                raise CVGError(msg)

            msg = '{} uncorrelated sample '.format(uncorrelated_samples_size)
            msg += 'points are not sufficient to be used by "UCL".'
            cvg_warning(msg)

        # compute mean
        self.mean = uncorrelated_samples.mean()

        # If the population standard deviation is unknown
        if population_standard_deviation is None:
            # Compute the sample standard deviation
            self.std = uncorrelated_samples.std()
        # If the population standard deviation is known
        else:
            self.std = population_standard_deviation

        # Compute the standard deviation of the mean within the dataset. The
        # standard_error_of_mean provides a measurement for spread. The smaller
        # the spread the more accurate.
        standard_error_of_mean = self.std / sqrt(uncorrelated_samples_size)

        # Compute the t_distribution confidence interval. When using the
        # t-distribution to compute a confidence interval, df = n - 1.
        p_up = (1 + confidence_coefficient) / 2
        upper = t_inv_cdf(p_up, uncorrelated_samples_size - 1)

        upper_confidence_limit = upper * standard_error_of_mean
        return upper_confidence_limit


def uncorrelated_samples_ucl(time_series_data,
                             *,
                             confidence_coefficient=0.95,
                             population_standard_deviation=None,
                             si=None,
                             fft=True,
                             minimum_correlation_time=None,
                             uncorrelated_sample_indices=None,
                             sample_method=None,
                             obj=None):
    """Approximate the upper confidence limit of the mean."""
    uncorrelated_samples = UncorrelatedSamples() if obj is None else obj
    upper_confidence_limit = uncorrelated_samples.ucl(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        population_standard_deviation=population_standard_deviation,
        si=si,
        fft=fft,
        minimum_correlation_time=minimum_correlation_time,
        uncorrelated_sample_indices=uncorrelated_sample_indices,
        sample_method=sample_method)
    return upper_confidence_limit


def uncorrelated_samples_ci(time_series_data,
                            *,
                            confidence_coefficient=0.95,
                            population_standard_deviation=None,
                            si=None,
                            fft=True,
                            minimum_correlation_time=None,
                            uncorrelated_sample_indices=None,
                            sample_method=None,
                            obj=None):
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
        sample_method=sample_method)
    return confidence_limits


def uncorrelated_samples_relative_half_width_estimate(
        time_series_data,
        *,
        confidence_coefficient=0.95,
        population_standard_deviation=None,
        si=None,
        fft=True,
        minimum_correlation_time=None,
        uncorrelated_sample_indices=None,
        sample_method=None,
        obj=None):
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
    relative_half_width_estimate = \
        uncorrelated_samples.relative_half_width_estimate(
            time_series_data=time_series_data,
            confidence_coefficient=confidence_coefficient,
            population_standard_deviation=population_standard_deviation,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            uncorrelated_sample_indices=uncorrelated_sample_indices,
            sample_method=sample_method)
    return relative_half_width_estimate
