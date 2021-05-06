"""Time series module."""

from .run_length_control import \
    run_length_control

from .utils import \
    time_series_data_si, \
    uncorrelated_time_series_data_sample_indices, \
    uncorrelated_time_series_data_samples, \
    time_series_data_uncorrelated_samples, \
    time_series_data_uncorrelated_random_samples, \
    time_series_data_uncorrelated_block_averaged_samples

__all__ = [
    'run_length_control',
    'time_series_data_si',
    'uncorrelated_time_series_data_sample_indices',
    'uncorrelated_time_series_data_samples',
    'time_series_data_uncorrelated_samples',
    'time_series_data_uncorrelated_random_samples',
    'time_series_data_uncorrelated_block_averaged_samples',
]
