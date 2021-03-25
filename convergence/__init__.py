r"""Convergence package."""

from .err import CVGError, cvg_warning
from .stats import \
    get_fft_optimal_size, \
    auto_covariance, \
    auto_correlate, \
    cross_covariance, \
    cross_correlate, \
    modified_periodogram, \
    periodogram
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
from .s_normal_dist import s_normal_inv_cdf
from .t_dist import t_inv_cdf, t_interval
from .statistical_inefficiency import \
    statistical_inefficiency, \
    r_statistical_inefficiency, \
    split_r_statistical_inefficiency,\
    split_statistical_inefficiency, \
    si_methods, \
    integrated_auto_correlation_time
from .utils import \
    validate_split, \
    train_test_split, \
    subsample, \
    subsample_index, \
    random_subsample, \
    block_average_subsample
from .equilibration_length import estimate_equilibration_length
from .ucl import \
    HeidelbergerWelch, \
    ucl, \
    subsamples_ucl
from .timeseries import \
    run_length_control

__all__ = [
    'CVGError',
    'cvg_warning',
    'get_fft_optimal_size',
    'auto_covariance',
    'auto_correlate',
    'cross_covariance',
    'cross_correlate',
    'modified_periodogram',
    'periodogram',
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
    's_normal_inv_cdf',
    't_inv_cdf',
    't_interval',
    'statistical_inefficiency',
    'r_statistical_inefficiency',
    'split_r_statistical_inefficiency',
    'split_statistical_inefficiency',
    'si_methods',
    'integrated_auto_correlation_time',
    'validate_split',
    'train_test_split',
    'subsample',
    'subsample_index',
    'random_subsample',
    'block_average_subsample',
    'estimate_equilibration_length',
    'HeidelbergerWelch',
    'ucl',
    'subsamples_ucl',
    'run_length_control',
]

__author__ = 'Yaser Afshar <yafshar@openkim.org>'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
