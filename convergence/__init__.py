r"""Convergence package."""

from .err import CVGError, cvg_warning
from .stats import \
    beta, \
    betacf, \
    betai, \
    betai_cdf_ccdf, \
    betai_cdf, \
    randomness_test, \
    s_normal_inv_cdf, \
    normal_inv_cdf, \
    normal_interval, \
    get_fft_optimal_size, \
    auto_covariance, \
    auto_correlate, \
    cross_covariance, \
    cross_correlate, \
    modified_periodogram, \
    periodogram, \
    int_power, \
    moment, \
    skew, \
    t_cdf_ccdf, \
    t_cdf, \
    t_inv_cdf, \
    t_interval, \
    ZERO_RC_BOUNDS, \
    ZERO_RC
from .outlier import \
    outlier_methods, \
    outlier_test
from .scale import \
    MinMaxScale, \
    minmax_scale, \
    TranslateScale, \
    translate_scale, \
    StandardScale, \
    standard_scale, \
    RobustScale, \
    robust_scale, \
    MaxAbsScale, \
    maxabs_scale, \
    scale_methods
from .batch import batch
from .mser_m import mser_m
#from .geweke import geweke
from .statistical_inefficiency import \
    statistical_inefficiency, \
    r_statistical_inefficiency, \
    split_r_statistical_inefficiency,\
    split_statistical_inefficiency, \
    si_methods, \
    integrated_auto_correlation_time
from .utils import \
    validate_split, \
    train_test_split
from .timeseries import \
    run_length_control, \
    time_series_data_si, \
    uncorrelated_time_series_data_sample_indices, \
    uncorrelated_time_series_data_samples, \
    time_series_data_uncorrelated_samples, \
    time_series_data_uncorrelated_random_samples, \
    time_series_data_uncorrelated_block_averaged_samples
from .equilibration_length import estimate_equilibration_length
from .ucl import \
    HeidelbergerWelch, \
    heidelberger_welch_ucl, \
    heidelberger_welch_ci, \
    heidelberger_welch_relative_half_width_estimate, \
    UncorrelatedSamples, \
    uncorrelated_samples_ucl, \
    uncorrelated_samples_ci, \
    uncorrelated_samples_relative_half_width_estimate, \
    N_SKART, \
    n_skart_ucl, \
    n_skart_ci, \
    n_skart_relative_half_width_estimate

__all__ = [
    'CVGError',
    'cvg_warning',
    'beta',
    'betacf',
    'betai',
    'betai_cdf_ccdf',
    'betai_cdf',
    'randomness_test',
    's_normal_inv_cdf',
    'normal_inv_cdf',
    'normal_interval',
    'get_fft_optimal_size',
    'auto_covariance',
    'auto_correlate',
    'cross_covariance',
    'cross_correlate',
    'modified_periodogram',
    'periodogram',
    'int_power',
    'moment',
    'skew',
    't_cdf_ccdf',
    't_cdf',
    't_inv_cdf',
    't_interval',
    'ZERO_RC_BOUNDS',
    'ZERO_RC',
    'outlier_methods',
    'outlier_test',
    'MinMaxScale',
    'minmax_scale',
    'TranslateScale',
    'translate_scale',
    'StandardScale',
    'standard_scale',
    'RobustScale',
    'robust_scale',
    'MaxAbsScale',
    'maxabs_scale',
    'scale_methods',
    'batch',
    # 'geweke',
    'mser_m',
    'statistical_inefficiency',
    'r_statistical_inefficiency',
    'split_r_statistical_inefficiency',
    'split_statistical_inefficiency',
    'si_methods',
    'integrated_auto_correlation_time',
    'validate_split',
    'train_test_split',
    'run_length_control',
    'time_series_data_si',
    'uncorrelated_time_series_data_sample_indices',
    'uncorrelated_time_series_data_samples',
    'time_series_data_uncorrelated_samples',
    'time_series_data_uncorrelated_random_samples',
    'time_series_data_uncorrelated_block_averaged_samples',
    'estimate_equilibration_length',
    'HeidelbergerWelch',
    'heidelberger_welch_ucl',
    'heidelberger_welch_ci',
    'heidelberger_welch_relative_half_width_estimate',
    'UncorrelatedSamples',
    'uncorrelated_samples_ucl',
    'uncorrelated_samples_ci',
    'uncorrelated_samples_relative_half_width_estimate',
    'N_SKART',
    'n_skart_ucl',
    'n_skart_ci',
    'n_skart_relative_half_width_estimate',
    'ucl_methods',
]

__author__ = 'Yaser Afshar <yafshar@openkim.org>'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
