"""UCL Base module."""

from math import fabs, isclose

from numpy.core.fromnumeric import std

from convergence import \
    CVGError, \
    estimate_equilibration_length, \
    time_series_data_si, \
    uncorrelated_time_series_data_sample_indices
from convergence._default import \
    __RELATIVE_HALF_WIDTH_ESTIMATE_ABS_TOL, \
    __CONFIDENCE_COEFFICIENT, \
    __EQUILIBRATION_LENGTH_ESTIMATE, \
    __HEIDEL_WELCH_NUMBER_POINTS, \
    __BATCH_SIZE, \
    __FFT, \
    __SCALE_METHOD, \
    __WITH_CENTERING, \
    __WITH_SCALING, \
    __TEST_SIZE, \
    __TRAIN_SIZE, \
    __POPULATION_STANDARD_DEVIATION, \
    __SI, \
    __MINIMUM_CORRELATION_TIME, \
    __UNCORRELATED_SAMPLE_INDICES, \
    __SAMPLE_METHOD, \
    __NSKIP, \
    __IGNORE_END

__all__ = [
    'UCLBase',
]


class UCLBase:
    """Upper Confidence Limit base class."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.indices_ = None
        self.si_ = None
        self.mean_ = None
        self.std_ = None

    @property
    def indices(self):
        """Get the indices."""
        return self.indices_

    @indices.setter
    def indices(self, value):
        """Set the indices.

        Args:
            value (1darray): indices array.

        """
        self.indices_ = value

    @indices.deleter
    def indices(self):
        """Delete the indices."""
        del self.indices_

    def set_indices(self,
                    time_series_data,
                    *,
                    si=__SI,
                    fft=__FFT,
                    minimum_correlation_time=__MINIMUM_CORRELATION_TIME):
        r"""Set the indices.

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

        """
        self.set_si(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time)

        si_value = self.si

        self.indices_ = uncorrelated_time_series_data_sample_indices(
            time_series_data=time_series_data,
            si=si_value,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time)

    @property
    def si(self):
        """Get the si."""
        return self.si_

    @si.setter
    def si(self, value):
        """Set the si (statistical inefficiency).

        Args:
            value (float): estimated statistical inefficiency value.

        """
        self.si_ = value

    @si.deleter
    def si(self):
        """Delete the si."""
        del self.si_

    def set_si(self,
               time_series_data,
               *,
               si=__SI,
               fft=__FFT,
               minimum_correlation_time=__MINIMUM_CORRELATION_TIME):
        r"""Set the si (statistical inefficiency).

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

        """
        self.si_ = time_series_data_si(
            time_series_data=time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time)

    @property
    def mean(self):
        """Get the mean."""
        return self.mean_

    @mean.setter
    def mean(self, value):
        """Set the mean.

        Args:
            value (float): mean value.

        """
        self.mean_ = value

    @mean.deleter
    def mean(self):
        """Delete the mean."""
        del self.mean_

    @property
    def std(self):
        """Get the std."""
        return self.std_

    @std.setter
    def std(self, value):
        """Set the std.

        Args:
            value (float): std value.

        """
        self.std_ = value

    @std.deleter
    def std(self):
        """Delete the std."""
        del self.std_

    def estimate_equilibration_length(
        self,
        time_series_data,
        *,
        si=__SI,
        nskip=__NSKIP,
        fft=__FFT,
        minimum_correlation_time=__MINIMUM_CORRELATION_TIME,
        ignore_end=__IGNORE_END,
        # unused input parmeters in Time series module
        # estimate_equilibration_length interface
        batch_size=__BATCH_SIZE,
        scale=__SCALE_METHOD,
        with_centering=__WITH_CENTERING,
        with_scaling=__WITH_SCALING):
        """Estimate the equilibration point in a time series data."""
        return estimate_equilibration_length(
            time_series_data=time_series_data,
            si=si,
            nskip=nskip,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            ignore_end=ignore_end)

    def ucl(self,
            time_series_data,
            *,
            confidence_coefficient=__CONFIDENCE_COEFFICIENT,
            equilibration_length_estimate=__EQUILIBRATION_LENGTH_ESTIMATE,
            heidel_welch_number_points=__HEIDEL_WELCH_NUMBER_POINTS,
            batch_size=__BATCH_SIZE,
            fft=__FFT,
            scale=__SCALE_METHOD,
            with_centering=__WITH_CENTERING,
            with_scaling=__WITH_SCALING,
            test_size=__TEST_SIZE,
            train_size=__TRAIN_SIZE,
            population_standard_deviation=__POPULATION_STANDARD_DEVIATION,
            si=__SI,
            minimum_correlation_time=__MINIMUM_CORRELATION_TIME,
            uncorrelated_sample_indices=__UNCORRELATED_SAMPLE_INDICES,
            sample_method=__SAMPLE_METHOD):
        """Approximate the upper confidence limit of the mean."""
        return 1e100

    def ci(self,
           time_series_data,
           *,
           confidence_coefficient=__CONFIDENCE_COEFFICIENT,
           equilibration_length_estimate=__EQUILIBRATION_LENGTH_ESTIMATE,
           heidel_welch_number_points=__HEIDEL_WELCH_NUMBER_POINTS,
           batch_size=__BATCH_SIZE,
           fft=__FFT,
           scale=__SCALE_METHOD,
           with_centering=__WITH_CENTERING,
           with_scaling=__WITH_SCALING,
           test_size=__TEST_SIZE,
           train_size=__TRAIN_SIZE,
           population_standard_deviation=__POPULATION_STANDARD_DEVIATION,
           si=__SI,
           minimum_correlation_time=__MINIMUM_CORRELATION_TIME,
           uncorrelated_sample_indices=__UNCORRELATED_SAMPLE_INDICES,
           sample_method=__SAMPLE_METHOD):
        """Approximate the confidence interval of the mean."""
        upper_confidence_limit = self.ucl(
            time_series_data=time_series_data,
            confidence_coefficient=confidence_coefficient,
            equilibration_length_estimate=equilibration_length_estimate,
            heidel_welch_number_points=heidel_welch_number_points,
            batch_size=batch_size,
            fft=fft,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
            test_size=test_size,
            train_size=train_size,
            population_standard_deviation=population_standard_deviation,
            si=si,
            minimum_correlation_time=minimum_correlation_time,
            uncorrelated_sample_indices=uncorrelated_sample_indices,
            sample_method=sample_method)
        lower_interval = self.mean - upper_confidence_limit
        upper_interval = self.mean + upper_confidence_limit
        return lower_interval, upper_interval

    def relative_half_width_estimate(
            self,
            time_series_data,
            *,
            confidence_coefficient=__CONFIDENCE_COEFFICIENT,
            equilibration_length_estimate=__EQUILIBRATION_LENGTH_ESTIMATE,
            heidel_welch_number_points=__HEIDEL_WELCH_NUMBER_POINTS,
            batch_size=__BATCH_SIZE,
            fft=__FFT,
            scale=__SCALE_METHOD,
            with_centering=__WITH_CENTERING,
            with_scaling=__WITH_SCALING,
            test_size=__TEST_SIZE,
            train_size=__TRAIN_SIZE,
            population_standard_deviation=__POPULATION_STANDARD_DEVIATION,
            si=__SI,
            minimum_correlation_time=__MINIMUM_CORRELATION_TIME,
            uncorrelated_sample_indices=__UNCORRELATED_SAMPLE_INDICES,
            sample_method=__SAMPLE_METHOD):
        """Get the relative half width estimate."""
        upper_confidence_limit = self.ucl(
            time_series_data=time_series_data,
            confidence_coefficient=confidence_coefficient,
            equilibration_length_estimate=equilibration_length_estimate,
            heidel_welch_number_points=heidel_welch_number_points,
            batch_size=batch_size,
            fft=fft,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
            test_size=test_size,
            train_size=train_size,
            population_standard_deviation=population_standard_deviation,
            si=si,
            minimum_correlation_time=minimum_correlation_time,
            uncorrelated_sample_indices=uncorrelated_sample_indices,
            sample_method=sample_method)

        # Estimat the relative half width
        if isclose(self.mean, 0,
                   abs_tol=__RELATIVE_HALF_WIDTH_ESTIMATE_ABS_TOL):
            msg = 'It is not possible to estimate the relative half width '
            msg += 'for the close to zero mean = {}'.format(self.mean)
            raise CVGError(msg)

        relative_half_width_estimate = upper_confidence_limit / fabs(self.mean)
        return relative_half_width_estimate
