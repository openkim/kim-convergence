r"""Convergence package."""

from .batch import batch
from .err import \
    CVGError, \
    CVGSampleSizeError, \
    cvg_warning, \
    cvg_check
#from .geweke import geweke
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
from .stats import \
    beta, \
    betacf, \
    betai, \
    betai_cdf_ccdf, \
    betai_cdf, \
    chi_square_test, \
    ContinuousDistributions, \
    ContinuousDistributionsNumberOfRequiredArguments, \
    ContinuousDistributionsArgumentRequirement, \
    check_population_cdf_args, \
    get_distribution_stats, \
    kruskal_test, \
    ks_test, \
    levene_test, \
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
    t_test, \
    ZERO_RC_BOUNDS, \
    ZERO_RC, \
    wilcoxon_test
from .timeseries import \
    estimate_equilibration_length, \
    statistical_inefficiency, \
    geyer_r_statistical_inefficiency, \
    geyer_split_r_statistical_inefficiency,\
    geyer_split_statistical_inefficiency, \
    si_methods, \
    integrated_auto_correlation_time, \
    time_series_data_si, \
    uncorrelated_time_series_data_sample_indices, \
    uncorrelated_time_series_data_samples, \
    time_series_data_uncorrelated_samples, \
    time_series_data_uncorrelated_random_samples, \
    time_series_data_uncorrelated_block_averaged_samples, \
    run_length_control
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
    n_skart_relative_half_width_estimate, \
    MSER_m, \
    mser_m_ucl, \
    mser_m_ci, \
    mser_m_relative_half_width_estimate, \
    mser_m,  \
    MSER_m_y, \
    mser_m_y_ucl, \
    mser_m_y_ci, \
    mser_m_y_relative_half_width_estimate, \
    ucl_methods
from .utils import \
    validate_split, \
    train_test_split


__all__ = [
    'CVGError',
    'CVGSampleSizeError',
    'cvg_warning',
    'cvg_check',
    # stats module
    'beta',
    'betacf',
    'betai',
    'betai_cdf_ccdf',
    'betai_cdf',
    'chi_square_test',
    'ContinuousDistributions',
    'ContinuousDistributionsNumberOfRequiredArguments',
    'ContinuousDistributionsArgumentRequirement',
    'check_population_cdf_args',
    'get_distribution_stats',
    'kruskal_test',
    'ks_test',
    'levene_test',
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
    't_test',
    'ZERO_RC_BOUNDS',
    'ZERO_RC',
    'wilcoxon_test',
    #
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
    'statistical_inefficiency',
    'geyer_r_statistical_inefficiency',
    'geyer_split_r_statistical_inefficiency',
    'geyer_split_statistical_inefficiency',
    'si_methods',
    'integrated_auto_correlation_time',
    'validate_split',
    'train_test_split',
    # time series module
    'estimate_equilibration_length',
    'run_length_control',
    'time_series_data_si',
    'uncorrelated_time_series_data_sample_indices',
    'uncorrelated_time_series_data_samples',
    'time_series_data_uncorrelated_samples',
    'time_series_data_uncorrelated_random_samples',
    'time_series_data_uncorrelated_block_averaged_samples',
    # ucl module
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
    'MSER_m',
    'mser_m_ucl',
    'mser_m_ci',
    'mser_m_relative_half_width_estimate',
    'mser_m',
    'MSER_m_y',
    'mser_m_y_ucl',
    'mser_m_y_ci',
    'mser_m_y_relative_half_width_estimate',
    'ucl_methods',
]

__author__ = 'Yaser Afshar <yafshar@openkim.org>'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
