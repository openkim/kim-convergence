r"""Convergence package."""

from .err import CVGError
from .stats import \
    get_fft_optimal_size, \
    auto_covariance, \
    auto_correlate, \
    cross_covariance, \
    cross_correlate, \
    translate_scale, \
    standard_scale, \
    robust_scale, \
    periodogram
from .batch import batch
from .mser_m import mser_m
#from .geweke import geweke
from .s_normal_dist import s_normal_inv_cdf
from .t_dist import t_inv_cdf
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
    subsample
from .equilibration_length import estimate_equilibration_length
from .ucl import \
    set_heidel_welch_constants, \
    get_heidel_welch_constants, \
    get_heidel_welch_set, \
    get_heidel_welch_knp, \
    get_heidel_welch_A, \
    get_heidel_welch_C1, \
    get_heidel_welch_C2, \
    get_heidel_welch_tm, \
    ucl
from .timeseries import \
    run_length_control

__all__ = [
    'CVGError',
    'get_fft_optimal_size',
    'auto_covariance',
    'auto_correlate',
    'cross_covariance',
    'cross_correlate',
    'translate_scale',
    'standard_scale',
    'robust_scale',
    'periodogram',
    'batch',
    # 'geweke',
    'mser_m',
    's_normal_inv_cdf',
    't_inv_cdf',
    'statistical_inefficiency',
    'r_statistical_inefficiency',
    'split_r_statistical_inefficiency',
    'split_statistical_inefficiency',
    'si_methods',
    'integrated_auto_correlation_time',
    'validate_split',
    'train_test_split',
    'subsample',
    'estimate_equilibration_length',
    'set_heidel_welch_constants',
    'get_heidel_welch_constants',
    'get_heidel_welch_set',
    'get_heidel_welch_knp',
    'get_heidel_welch_A',
    'get_heidel_welch_C1',
    'get_heidel_welch_C2',
    'get_heidel_welch_tm',
    'ucl',
    'run_length_control',
]

__author__ = 'Yaser Afshar <yafshar@umn.edu>'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
