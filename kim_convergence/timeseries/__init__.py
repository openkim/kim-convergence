"""Time series module."""

from .equilibration_length import estimate_equilibration_length
from .statistical_inefficiency import \
    geyer_r_statistical_inefficiency, \
    geyer_split_r_statistical_inefficiency, \
    geyer_split_statistical_inefficiency, \
    integrated_auto_correlation_time, \
    statistical_inefficiency, \
    si_methods
from .utils import \
    time_series_data_si, \
    time_series_data_uncorrelated_samples, \
    time_series_data_uncorrelated_random_samples, \
    time_series_data_uncorrelated_block_averaged_samples, \
    uncorrelated_time_series_data_samples, \
    uncorrelated_time_series_data_sample_indices


__all__ = [
    # equilibration_length
    'estimate_equilibration_length',
    # statistical_inefficiency
    'geyer_r_statistical_inefficiency',
    'geyer_split_r_statistical_inefficiency',
    'geyer_split_statistical_inefficiency',
    'integrated_auto_correlation_time',
    'si_methods',
    'statistical_inefficiency',
    # utils
    'time_series_data_si',
    'time_series_data_uncorrelated_samples',
    'time_series_data_uncorrelated_random_samples',
    'time_series_data_uncorrelated_block_averaged_samples',
    'uncorrelated_time_series_data_samples',
    'uncorrelated_time_series_data_sample_indices',
]
