"""UCL Base module."""

from math import fabs, isclose
import numpy as np
from typing import Optional, Union, List

from kim_convergence._default import \
    _DEFAULT_RELATIVE_HALF_WIDTH_ESTIMATE_ABS_TOL, \
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
    _DEFAULT_SAMPLE_METHOD, \
    _DEFAULT_NSKIP, \
    _DEFAULT_IGNORE_END, \
    _DEFAULT_NUMBER_OF_CORES
from kim_convergence import \
    CRError, \
    estimate_equilibration_length, \
    time_series_data_si, \
    uncorrelated_time_series_data_sample_indices


__all__ = [
    'UCLBase',
]


class UCLBase:
    """Upper Confidence Limit base class."""

    def __init__(self):
        """Constructor."""
        self.reset()

    def reset(self) -> None:
        self.name_ = 'base'
        self.indices_ = None
        self.si_ = None
        self.mean_ = None
        self.std_ = None
        self.sample_size_ = None
        self.upper_confidence_limit = None

    @property
    def name(self):
        """Get the name."""
        return self.name_

    @name.setter
    def name(self, value):
        """Set the name.

        Args:
            value (str): name.

        """
        self.name_ = value

    @name.deleter
    def name(self):
        """Delete the name."""
        del self.name_

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

    def set_indices(
            self,
            time_series_data: Union[np.ndarray, List[float]],
            *,
            si: Union[str, float, int, None] = _DEFAULT_SI,
            fft: bool = _DEFAULT_FFT,
            minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME) -> None:
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

        self.indices = uncorrelated_time_series_data_sample_indices(
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

    def set_si(
            self,
            time_series_data,
            *,
            si: Union[str, float, int, None] = _DEFAULT_SI,
            fft: bool = _DEFAULT_FFT,
            minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME) -> None:
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
        self.si = time_series_data_si(
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

    @property
    def sample_size(self):
        """Get the sample_size."""
        return self.sample_size_

    @sample_size.setter
    def sample_size(self, value):
        """Set the sample_size.

        Args:
            value (int): sample_size value.

        """
        self.sample_size_ = value

    @sample_size.deleter
    def sample_size(self):
        """Delete the sample_size."""
        del self.sample_size_

    def estimate_equilibration_length(
            self,
            time_series_data: Union[np.ndarray, List[float]],
            *,
            si: Union[str, float, int, None] = _DEFAULT_SI,
            nskip: Optional[int] = _DEFAULT_NSKIP,
            fft: bool = _DEFAULT_FFT,
            minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
            ignore_end: Union[int, float, None] = _DEFAULT_IGNORE_END,
            number_of_cores: int = _DEFAULT_NUMBER_OF_CORES,
            # unused input parmeters in
            # estimate_equilibration_length interface
            batch_size: int = _DEFAULT_BATCH_SIZE,
            scale: str = _DEFAULT_SCALE_METHOD,
            with_centering: bool = _DEFAULT_WITH_CENTERING,
            with_scaling: bool = _DEFAULT_WITH_SCALING) -> tuple((bool, int)):
        """Estimate the equilibration point in a time series data."""
        equilibration_index_estimate, si_value = estimate_equilibration_length(
            time_series_data=time_series_data,
            si=si,
            nskip=nskip,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            ignore_end=ignore_end,
            number_of_cores=number_of_cores)

        if equilibration_index_estimate < len(time_series_data) - 1:
            self.si = si_value
            return True, equilibration_index_estimate

        self.si = None
        return False, equilibration_index_estimate

    def ucl(self,
            time_series_data: Union[np.ndarray, List[float]],
            *,
            confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
            equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
            heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
            batch_size: int = _DEFAULT_BATCH_SIZE,
            fft: bool = _DEFAULT_FFT,
            scale: str = _DEFAULT_SCALE_METHOD,
            with_centering: bool = _DEFAULT_WITH_CENTERING,
            with_scaling: bool = _DEFAULT_WITH_SCALING,
            test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
            train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
            population_standard_deviation: Optional[float] = _DEFAULT_POPULATION_STANDARD_DEVIATION,
            si: Union[str, float, int, None] = _DEFAULT_SI,
            minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
            uncorrelated_sample_indices: Union[np.ndarray, List[int],
                                               None] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
            sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD) -> float:
        """Approximate the upper confidence limit of the mean."""
        return 1e100

    def ci(self,
           time_series_data: Union[np.ndarray, List[float]],
           *,
           confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
           equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
           heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
           batch_size: int = _DEFAULT_BATCH_SIZE,
           fft: bool = _DEFAULT_FFT,
           scale: str = _DEFAULT_SCALE_METHOD,
           with_centering: bool = _DEFAULT_WITH_CENTERING,
           with_scaling: bool = _DEFAULT_WITH_SCALING,
           test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
           train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
           population_standard_deviation: Optional[float] = _DEFAULT_POPULATION_STANDARD_DEVIATION,
           si: Union[str, float, int, None] = _DEFAULT_SI,
           minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
           uncorrelated_sample_indices: Union[np.ndarray, List[int],
                                              None] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
           sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD) -> tuple((float, float)):
        """Approximate the confidence interval of the mean."""
        self.upper_confidence_limit = self.ucl(
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
        lower_interval = self.mean - self.upper_confidence_limit
        upper_interval = self.mean + self.upper_confidence_limit
        return lower_interval, upper_interval

    def relative_half_width_estimate(
            self,
            time_series_data: Union[np.ndarray, List[float]],
            *,
            confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
            equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
            heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
            batch_size: int = _DEFAULT_BATCH_SIZE,
            fft: bool = _DEFAULT_FFT,
            scale: str = _DEFAULT_SCALE_METHOD,
            with_centering: bool = _DEFAULT_WITH_CENTERING,
            with_scaling: bool = _DEFAULT_WITH_SCALING,
            test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
            train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
            population_standard_deviation: Optional[float] = _DEFAULT_POPULATION_STANDARD_DEVIATION,
            si: Union[str, float, int, None] = _DEFAULT_SI,
            minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
            uncorrelated_sample_indices: Union[np.ndarray, List[int],
                                               None] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
            sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD) -> float:
        """Get the relative half width estimate."""
        self.upper_confidence_limit = self.ucl(
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
                   abs_tol=_DEFAULT_RELATIVE_HALF_WIDTH_ESTIMATE_ABS_TOL):
            msg = 'It is not possible to estimate the relative half width '
            msg += 'for the close to zero mean = {}'.format(self.mean)
            raise CRError(msg)

        relative_half_width_estimate = \
            self.upper_confidence_limit / fabs(self.mean)
        return relative_half_width_estimate
