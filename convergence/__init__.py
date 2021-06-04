r"""Convergence package."""

from .batch import batch
#from .geweke import geweke
from ._default import *
from .err import \
    CVGError, \
    CVGSampleSizeError, \
    cvg_warning, \
    cvg_check
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
    s_normal_inv_cdf, \
    normal_inv_cdf, \
    normal_interval, \
    chi_square_test, \
    t_test, \
    ContinuousDistributions, \
    ContinuousDistributionsNumberOfRequiredArguments, \
    ContinuousDistributionsArgumentRequirement, \
    check_population_cdf_args, \
    get_distribution_stats, \
    levene_test, \
    kruskal_test, \
    ks_test, \
    wilcoxon_test, \
    randomness_test, \
    t_cdf_ccdf, \
    t_cdf, \
    t_inv_cdf, \
    t_interval, \
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
    ZERO_RC_BOUNDS, \
    ZERO_RC
from .utils import \
    validate_split, \
    train_test_split
from .timeseries import \
    estimate_equilibration_length, \
    geyer_r_statistical_inefficiency, \
    geyer_split_r_statistical_inefficiency, \
    geyer_split_statistical_inefficiency, \
    integrated_auto_correlation_time, \
    si_methods, \
    statistical_inefficiency, \
    time_series_data_si, \
    time_series_data_uncorrelated_samples, \
    time_series_data_uncorrelated_random_samples, \
    time_series_data_uncorrelated_block_averaged_samples, \
    uncorrelated_time_series_data_samples, \
    uncorrelated_time_series_data_sample_indices
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
from .run_length_control import run_length_control


__all__ = [
    # batch module
    'batch',
    # 'geweke',
    # err module
    'CVGError',
    'CVGSampleSizeError',
    'cvg_warning',
    'cvg_check',
    # outlier module
    'outlier_methods',
    'outlier_test',
    # scale module
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
    # stats module
    'beta',
    'betacf',
    'betai',
    'betai_cdf_ccdf',
    'betai_cdf',
    's_normal_inv_cdf',
    'normal_inv_cdf',
    'normal_interval',
    'chi_square_test',
    't_test',
    'ContinuousDistributions',
    'ContinuousDistributionsNumberOfRequiredArguments',
    'ContinuousDistributionsArgumentRequirement',
    'check_population_cdf_args',
    'get_distribution_stats',
    'levene_test',
    'kruskal_test',
    'ks_test',
    'wilcoxon_test'
    'randomness_test',
    't_cdf_ccdf',
    't_cdf',
    't_inv_cdf',
    't_interval',
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
    'ZERO_RC_BOUNDS',
    'ZERO_RC',
    # utils module
    'validate_split',
    'train_test_split',
    # time series module
    'estimate_equilibration_length',
    'geyer_r_statistical_inefficiency',
    'geyer_split_r_statistical_inefficiency',
    'geyer_split_statistical_inefficiency',
    'integrated_auto_correlation_time',
    'si_methods',
    'statistical_inefficiency',
    'time_series_data_si',
    'time_series_data_uncorrelated_samples',
    'time_series_data_uncorrelated_random_samples',
    'time_series_data_uncorrelated_block_averaged_samples',
    'uncorrelated_time_series_data_samples',
    'uncorrelated_time_series_data_sample_indices',
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
    # run_length_control module
    'run_length_control',
]


__author__ = 'Yaser Afshar <yafshar@openkim.org>'


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
