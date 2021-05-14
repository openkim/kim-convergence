"""Time series module."""

from .equilibration_length import \
    estimate_equilibration_length

# from .run_length_control import \
#     run_length_control

from .statistical_inefficiency import \
    statistical_inefficiency, \
    r_statistical_inefficiency, \
    split_r_statistical_inefficiency,\
    split_statistical_inefficiency, \
    si_methods, \
    integrated_auto_correlation_time

from .utils import \
    time_series_data_si, \
    uncorrelated_time_series_data_sample_indices, \
    uncorrelated_time_series_data_samples, \
    time_series_data_uncorrelated_samples, \
    time_series_data_uncorrelated_random_samples, \
    time_series_data_uncorrelated_block_averaged_samples

__all__ = [
    'estimate_equilibration_length',
    # 'run_length_control',
    'statistical_inefficiency',
    'r_statistical_inefficiency',
    'split_r_statistical_inefficiency',
    'split_statistical_inefficiency',
    'si_methods',
    'integrated_auto_correlation_time',
    'time_series_data_si',
    'uncorrelated_time_series_data_sample_indices',
    'uncorrelated_time_series_data_samples',
    'time_series_data_uncorrelated_samples',
    'time_series_data_uncorrelated_random_samples',
    'time_series_data_uncorrelated_block_averaged_samples',
]
