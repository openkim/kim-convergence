"""Test module for non-normally distributed data.

Note:
    The tests in this module are modified and fixed for the convergence
    package use.

"""
import numpy as np
from scipy.stats import distributions, kruskal, kstest, levene, wilcoxon

from convergence._default import _DEFAULT_CONFIDENCE_COEFFICIENT
from convergence import CVGError, CVGSampleSizeError, cvg_check


__all__ = [
    'ContinuousDistributions',
    'ContinuousDistributionsNumberOfRequiredArguments',
    'ContinuousDistributionsArgumentRequirement',
    'check_population_cdf_args',
    'ks_test',
    'levene_test',
    'wilcoxon_test',
    'kruskal_test',
]

"""Continuous distributions.

References:
    .. [22] https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

"""
ContinuousDistributions = {
    'alpha': 'Alpha distribution',
    'anglit': 'Anglit distribution',
    'arcsine': 'Arcsine distribution',
    'argus': 'Argus distribution',
    'beta': 'Beta distribution',
    'betaprime': 'Beta prime distribution',
    'bradford': 'Bradford distribution',
    'burr': 'Burr (Type III) distribution',
    'burr12': 'Burr (Type XII) distribution',
    'cauchy': 'Cauchy distribution',
    'chi': 'Chi distribution',
    'chi-squared': 'Chi-squared distribution',
    'cosine': 'Cosine distribution',
    'crystalball': 'Crystalball distribution',
    'dgamma': 'Double gamma distribution',
    'dweibull': 'Double Weibull distribution',
    'erlang': 'Erlang distribution',
    'expon': 'Exponential distribution',
    'exponnorm': 'Exponentially modified Normal distribution',
    'exponweib': 'Exponentiated Weibull distribution',
    'exponpow': 'Exponential power distribution',
    'f': 'F distribution',
    'fatiguelife': 'Fatigue-life (Birnbaum-Saunders) distribution',
    'fisk': 'Fisk distribution',
    'foldcauchy': 'Folded Cauchy distribution',
    'foldnorm': 'Folded normal distribution',
    'genlogistic': 'Generalized logistic distribution',
    'gennorm': 'Generalized normal distribution',
    'genpareto': 'Generalized Pareto distribution',
    'genexpon': 'Generalized exponential distribution',
    'genextreme': 'Generalized extreme value distribution',
    'gausshyper': 'Gauss hypergeometric distribution',
    'gamma': 'Gamma distribution',
    'gengamma': 'Generalized gamma distribution',
    'genhalflogistic': 'Generalized half-logistic distribution',
    'geninvgauss': 'Generalized Inverse Gaussian distribution',
    'gilbrat': 'Gilbrat distribution',
    'gompertz': 'Gompertz (or truncated Gumbel) distribution',
    'gumbel_r': 'Right-skewed Gumbel distribution',
    'gumbel_l': 'Left-skewed Gumbel distribution',
    'halfcauchy': 'Half-Cauchy distribution',
    'halflogistic': 'Half-logistic distribution',
    'halfnorm': 'Half-normal distribution',
    'halfgennorm': 'Upper half of a generalized normal distribution',
    'hypsecant': 'Hyperbolic secant distribution',
    'invgamma': 'Inverted gamma distribution',
    'invgauss': 'Inverse Gaussian distribution',
    'invweibull': 'Inverted Weibull distribution',
    'johnsonsb': 'Johnson SB distribution',
    'johnsonsu': 'Johnson SU distribution',
    'kappa4': 'Kappa 4 parameter distribution',
    'kappa3': 'Kappa 3 parameter distribution',
    'ksone': 'Kolmogorov-Smirnov one-sided test statistic distribution',
    'kstwo': 'Kolmogorov-Smirnov two-sided test statistic distribution',
    'kstwobign': 'Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic',
    'laplace': 'Laplace distribution',
    'laplace_asymmetric': 'Asymmetric Laplace distribution',
    'levy': 'Levy distribution',
    'levy_l': 'Left-skewed Levy distribution',
    'levy_stable': 'Levy-stable distribution',
    'logistic': 'Logistic (or Sech-squared) distribution',
    'loggamma': 'Log gamma distribution',
    'loglaplace': 'Log-Laplace distribution',
    'lognorm': 'Lognormal distribution',
    'loguniform': 'Loguniform or reciprocal distribution',
    'lomax': 'Lomax (Pareto of the second kind) distribution',
    'maxwell': 'Maxwell distribution',
    'mielke': 'Mielke Beta-Kappa / Dagum distribution',
    'moyal': 'Moyal distribution',
    'nakagami': 'Nakagami distribution',
    'ncx2': 'Non-central chi-squared distribution',
    'ncf': 'Non-central F distribution distribution',
    'nct': 'Non-central Student’s t distribution',
    'norm': 'Normal distribution',
    'norminvgauss': 'Normal Inverse Gaussian distribution',
    'pareto': 'Pareto distribution',
    'pearson3': 'Pearson type III distribution',
    'powerlaw': 'Power-function distribution',
    'powerlognorm': 'Power log-normal distribution',
    'powernorm': 'Power normal distribution',
    'rdist': 'R-distributed (symmetric beta) distribution',
    'rayleigh': 'Rayleigh distribution',
    'rice': 'Rice distribution',
    'recipinvgauss': 'Reciprocal inverse Gaussian distribution',
    'semicircular': 'Semicircular distribution',
    'skewnorm': 'Skew-normal distribution',
    't': 'Student’s t distribution',
    'trapezoid': 'Trapezoidal distribution',
    'triang': 'Triangular distribution',
    'truncexpon': 'Truncated exponential distribution',
    'truncnorm': 'Truncated normal distribution',
    'tukeylambda': 'Tukey-Lamdba distribution',
    'uniform': 'Uniform distribution',
    'vonmises': 'Von Mises distribution',
    'vonmises_line': 'Von Mises distribution',
    'wald': 'Wald distribution',
    'weibull_min': 'Weibull minimum distribution',
    'weibull_max': 'Weibull maximum distribution',
    'wrapcauchy': 'Wrapped Cauchy distribution',
}


"""Continuous distributions number of required arguments."""
ContinuousDistributionsNumberOfRequiredArguments = {
    'alpha': 1,
    'anglit': 0,
    'arcsine': 0,
    'argus': 1,
    'beta': 2,
    'betaprime': 2,
    'bradford': 1,
    'burr': 2,
    'burr12': 2,
    'cauchy': 0,
    'chi': 1,
    'chi-squared': 1,
    'cosine': 0,
    'crystalball': 2,
    'dgamma': 1,
    'dweibull': 1,
    'erlang': 1,
    'expon': 0,
    'exponnorm': 1,
    'exponweib': 2,
    'exponpow': 1,
    'f': 2,
    'fatiguelife': 1,
    'fisk': 1,
    'foldcauchy': 1,
    'foldnorm': 1,
    'genlogistic': 1,
    'gennorm': 1,
    'genpareto': 1,
    'genexpon': 3,
    'genextreme': 1,
    'gausshyper': 4,
    'gamma': 1,
    'gengamma': 2,
    'genhalflogistic': 1,
    'geninvgauss': 2,
    'gilbrat': 0,
    'gompertz': 1,
    'gumbel_r': 0,
    'gumbel_l': 0,
    'halfcauchy': 0,
    'halflogistic': 0,
    'halfnorm': 0,
    'halfgennorm': 1,
    'hypsecant': 0,
    'invgamma': 1,
    'invgauss': 1,
    'invweibull': 1,
    'johnsonsb': 2,
    'johnsonsu': 2,
    'kappa4': 2,
    'kappa3': 1,
    'ksone': 1,
    'kstwo': 1,
    'kstwobign': 0,
    'laplace': 0,
    'laplace_asymmetric': 1,
    'levy': 0,
    'levy_l': 0,
    'levy_stable': 2,
    'logistic': 0,
    'loggamma': 1,
    'loglaplace': 1,
    'lognorm': 1,
    'loguniform': 2,
    'lomax': 1,
    'maxwell': 0,
    'mielke': 2,
    'moyal': 0,
    'nakagami': 1,
    'ncx2': 2,
    'ncf': 3,
    'nct': 2,
    'norm': 0,
    'norminvgauss': 2,
    'pareto': 1,
    'pearson3': 3,
    'powerlaw': 1,
    'powerlognorm': 2,
    'powernorm': 1,
    'rdist': 1,
    'rayleigh': 0,
    'rice': 1,
    'recipinvgauss': 1,
    'semicircular': 0,
    'skewnorm': 1,
    't': 1,
    'trapezoid': 2,
    'triang': 1,
    'truncexpon': 1,
    'truncnorm': 2,
    'tukeylambda': 1,
    'uniform': 0,
    'vonmises': 1,
    'vonmises_line': 1,
    'wald': 0,
    'weibull_min': 1,
    'weibull_max': 1,
    'wrapcauchy': 1,
}


"""Continuous distributions argument requirement."""
ContinuousDistributionsArgumentRequirement = {
    'alpha': ContinuousDistributions['alpha'] + ' takes `a` as a shape parameter.',
    'anglit': ContinuousDistributions['anglit'] + ' takes no arguments.',
    'arcsine': ContinuousDistributions['arcsine'] + ' takes no arguments.',
    'argus': ContinuousDistributions['argus'] + ' takes `chi` as a shape parameter.',
    'beta': ContinuousDistributions['beta'] + ' takes `a` and `b` as shape parameters.',
    'betaprime': ContinuousDistributions['betaprime'] + ' takes `a` and `b` as shape parameters.',
    'bradford': ContinuousDistributions['bradford'] + ' takes `c` as a shape parameter.',
    'burr': ContinuousDistributions['burr'] + ' takes takes `c` and `d` as shape parameters.',
    'burr12': ContinuousDistributions['burr12'] + ' takes takes `c` and `d` as shape parameters.',
    'cauchy': ContinuousDistributions['cauchy'] + ' takes no arguments.',
    'chi': ContinuousDistributions['chi'] + ' takes `df` as a shape parameter.',
    'chi-squared': ContinuousDistributions['chi-squared'] + ' takes `df` as a shape parameter.',
    'cosine': ContinuousDistributions['cosine'] + ' takes no arguments.',
    'crystalball': ContinuousDistributions['crystalball'] + ' takes `beta > 0` and `m > 1` as shape parameters.',
    'dgamma': ContinuousDistributions['dgamma'] + ' takes `a` as a shape parameter.',
    'dweibull': ContinuousDistributions['dweibull'] + ' takes `c` as a shape parameter.',
    'erlang': ContinuousDistributions['erlang'] + ' takes `a` as a shape parameter.',
    'expon': ContinuousDistributions['expon'] + ' takes no arguments.',
    'exponnorm': ContinuousDistributions['exponnorm'] + ' takes :math:`K > 0` as a rate equals to :math:`1/K`.',
    'exponweib': ContinuousDistributions['exponweib'] + ' takes `a` and `c` as shape parameters.',
    'exponpow': ContinuousDistributions['exponpow'] + ' takes `b` as a shape parameter.',
    'f': ContinuousDistributions['f'] + ' takes `dfn` and `dfd` as shape parameters.',
    'fatiguelife': ContinuousDistributions['fatiguelife'] + ' takes `c` as a shape parameter.',
    'fisk': ContinuousDistributions['fisk'] + ' takes `c` as a shape parameter.',
    'foldcauchy': ContinuousDistributions['foldcauchy'] + ' takes `c` as a shape parameter.',
    'foldnorm': ContinuousDistributions['foldnorm'] + ' takes `c` as a shape parameter.',
    'genlogistic': ContinuousDistributions['genlogistic'] + ' takes `c` as a shape parameter.',
    'gennorm': ContinuousDistributions['gennorm'] + ' takes `beta` as a shape parameter.',
    'genpareto': ContinuousDistributions['genpareto'] + ' takes `c` as a shape parameter.',
    'genexpon': ContinuousDistributions['genexpon'] + ' takes `a` and `b` and `c` as shape parameters.',
    'genextreme': ContinuousDistributions['genextreme'] + ' takes `c` as a shape parameter.',
    'gausshyper': ContinuousDistributions['gausshyper'] + ' takes`a` and `b` and `c` and `z` as shape parameters.',
    'gamma': ContinuousDistributions['gamma'] + ' takes `a` as a shape parameter.',
    'gengamma': ContinuousDistributions['gengamma'] + ' takes`a` and `c` as shape parameters.',
    'genhalflogistic': ContinuousDistributions['genhalflogistic'] + ' takes`c` as a shape parameter.',
    'geninvgauss': ContinuousDistributions['geninvgauss'] + ' takes`p` and `b > 0` parameters.',
    'gilbrat': ContinuousDistributions['gilbrat'] + ' takesno arguments.',
    'gompertz': ContinuousDistributions['gompertz'] + ' takes`c` as a shape parameter.',
    'gumbel_r': ContinuousDistributions['gumbel_r'] + ' takes no arguments.',
    'gumbel_l': ContinuousDistributions['gumbel_l'] + ' takes no arguments.',
    'halfcauchy': ContinuousDistributions['halfcauchy'] + ' takes no arguments.',
    'halflogistic': ContinuousDistributions['halflogistic'] + ' takes no arguments.',
    'halfnorm': ContinuousDistributions['halfnorm'] + ' takes no arguments.',
    'halfgennorm': ContinuousDistributions['halfgennorm'] + ' takes `beta` as a shape parameter.',
    'hypsecant': ContinuousDistributions['hypsecant'] + ' takes no arguments.',
    'invgamma': ContinuousDistributions['invgamma'] + ' takes `a` as a shape parameter.',
    'invgauss': ContinuousDistributions['invgauss'] + ' takes `\mu` as a shape parameter.',
    'invweibull': ContinuousDistributions['invweibull'] + ' takes `c` as a shape parameter.',
    'johnsonsb': ContinuousDistributions['johnsonsb'] + ' takes `a` and `b` as shape parameters.',
    'johnsonsu': ContinuousDistributions['johnsonsu'] + ' takes `a` and `b` as shape parameters.',
    'kappa4': ContinuousDistributions['kappa4'] + ' takes `h` and `k` as shape parameters.',
    'kappa3': ContinuousDistributions['kappa3'] + ' takes `a` as a shape parameter.',
    'ksone': ContinuousDistributions['ksone'] + ' takes `n` as a shape parameter.',
    'kstwo': ContinuousDistributions['kstwo'] + ' takes `n` as a shape parameter.',
    'kstwobign': ContinuousDistributions['kstwobign'] + ' takes no arguments.',
    'laplace': ContinuousDistributions['laplace'] + ' takes no arguments.',
    'laplace_asymmetric': ContinuousDistributions['laplace_asymmetric'] + ' takes `kappa` as a shape parameter.',
    'levy': ContinuousDistributions['levy'] + ' takes no arguments.',
    'levy_l': ContinuousDistributions['levy_l'] + ' takes no arguments.',
    'levy_stable': ContinuousDistributions['levy_stable'] + ' takes `alpha` and `beta` parameters.',
    'logistic': ContinuousDistributions['logistic'] + ' takes no arguments.',
    'loggamma': ContinuousDistributions['loggamma'] + ' takes `c` as a shape parameter.',
    'loglaplace': ContinuousDistributions['loglaplace'] + ' takes `c` as a shape parameter.',
    'lognorm': ContinuousDistributions['lognorm'] + ' takes `s` as a shape parameter.',
    'loguniform': ContinuousDistributions['loguniform'] + ' takes `a` and `b` parameters.',
    'lomax': ContinuousDistributions['lomax'] + ' takes `c` as a shape parameter.',
    'maxwell': ContinuousDistributions['maxwell'] + ' takes no arguments.',
    'mielke': ContinuousDistributions['mielke'] + ' takes `k` and `s` as shape parameters.',
    'moyal': ContinuousDistributions['moyal'] + ' takes no arguments.',
    'nakagami': ContinuousDistributions['nakagami'] + ' takes `n` as a shape parameter.',
    'ncx2': ContinuousDistributions['ncx2'] + ' takes `df` and `nc` as shape parameters.',
    'ncf': ContinuousDistributions['ncf'] + ' takes `dfn` and `dfd` and `nc` as shape parameters.',
    'nct': ContinuousDistributions['nct'] + ' takes `df` and `nc` as shape parameters.',
    'norm': ContinuousDistributions['norm'] + ' takes no arguments.',
    'norminvgauss': ContinuousDistributions['norminvgauss'] + ' takes `a` and `b` as parameters.',
    'pareto': ContinuousDistributions['pareto'] + ' takes `b` as a shape parameter.',
    'pearson3': ContinuousDistributions['pearson3'] + ' takes `skew` as a shape parameter.',
    'powerlaw': ContinuousDistributions['powerlaw'] + ' takes `a` as a shape parameter.',
    'powerlognorm': ContinuousDistributions['powerlognorm'] + ' takes `c` and `s` as shape parameters.',
    'powernorm': ContinuousDistributions['powernorm'] + ' takes `c` as a shape parameter.',
    'rdist': ContinuousDistributions['rdist'] + ' takes `c` as a shape parameter.',
    'rayleigh': ContinuousDistributions['rayleigh'] + ' takes no arguments.',
    'rice': ContinuousDistributions['rice'] + ' takes `b` as a shape parameter.',
    'recipinvgauss': ContinuousDistributions['recipinvgauss'] + ' takes `mu` as a shape parameter.',
    'semicircular': ContinuousDistributions['semicircular'] + ' takes no arguments.',
    'skewnorm': ContinuousDistributions['skewnorm'] + ' takes `a` as a parameter.',
    't': ContinuousDistributions['t'] + ' takes `df` as a parameter.',
    'trapezoid': ContinuousDistributions['trapezoid'] + ' takes `c` and `d` as shape parameters.',
    'triang': ContinuousDistributions['triang'] + ' takes `c` as a shape parameter.',
    'truncexpon': ContinuousDistributions['truncexpon'] + ' takes `b` as a shape parameter.',
    'truncnorm': ContinuousDistributions['truncnorm'] + ' takes `a` and `b` as shape parameters.',
    'tukeylambda': ContinuousDistributions['tukeylambda'] + ' takes `lambda` as a shape parameter.',
    'uniform': ContinuousDistributions['uniform'] + ' takes no arguments.',
    'vonmises': ContinuousDistributions['vonmises'] + ' takes `kappa` as a shape parameter.',
    'vonmises_line': ContinuousDistributions['vonmises_line'] + ' takes `kappa` as a shape parameter.',
    'wald': ContinuousDistributions['wald'] + ' takes no arguments.',
    'weibull_min': ContinuousDistributions['weibull_min'] + ' takes `c` as a shape parameter.',
    'weibull_max': ContinuousDistributions['weibull_max'] + ' takes `c` as a shape parameter.',
    'wrapcauchy': ContinuousDistributions['wrapcauchy'] + ' takes `c` as a shape parameter.',
}


def check_population_cdf_args(population_cdf: str, population_args: tuple):
    """Check the input population_cdf and population_args for correctness.

    Args:
        population_cdf (str): The name of a distribution.
        population_args (tuple): Distribution parameter.

    """
    if population_cdf in ('default', None):
        return

    if population_cdf not in ContinuousDistributions:
        msg = 'The {} distribution is not supported.'.format(population_cdf)
        msg += 'It should be the name of a distribution in:\n'
        msg += '    https://docs.scipy.org/doc/scipy/reference/stats.html#'
        msg += 'continuous-distributions'
        raise CVGError(msg)

    number_of_required_arguments = \
        ContinuousDistributionsNumberOfRequiredArguments[population_cdf]
    number_of_arguments = len(population_args)

    if number_of_required_arguments != number_of_arguments:
        msg = 'The {} distribution requires '.format(population_cdf)
        if number_of_required_arguments in (0, 1):
            msg += '{} argument, but '.format(number_of_required_arguments)
            if number_of_arguments == 0:
                msg += 'no input argument is provided.'
            else:
                msg += '1 input argument is provided.'
        else:
            msg += '{} arguments, but '.format(number_of_required_arguments)
            msg += '{} input arguments '.format(number_of_arguments)
            msg += 'are provided.'

        Reference = '    https://docs.scipy.org/doc/scipy/reference/generated/'
        Reference += 'scipy.stats.{}.html#scipy.stats.'.format(population_cdf)
        Reference += '{}'.format(population_cdf)

        msg += '\n'
        msg += ContinuousDistributionsArgumentRequirement[population_cdf]
        msg += '\nReference:\n'
        msg += Reference
        raise CVGError(msg)


def ks_test(time_series_data: np.ndarray,
            population_cdf: str,
            population_args: tuple,
            population_loc: float,
            population_scale: float,
            significance_level=1 - _DEFAULT_CONFIDENCE_COEFFICIENT) -> bool:
    """Kolmogorov-Smirnov test for goodness of fit.

    Note:
        This test is only valid for continuous distributions.

    It uses the distribution of an observed variable against a given
    distribution.

    The null hypothesis is that the observed samples are drawn from the same
    continuous distribution as the given distribution with `population_loc`
    and `population_scale` if they are given.

    Note:
        The alternative hypothesis is `two-sided`. Where the empirical
        cumulative distribution function of the observed variables is `less` or
        `greater than the cumulative distribution function of the given
        distribution.

    The probability density of the given population distribution is in the
    `standardized` form. Thus to shift and/or scale the distribution
    `population_loc` and `population_scale` parameters are used.
    In these cases, the variable change `y <- x`, where `y = (x - loc) / scale`

    Args:
        time_series_data (np.ndarray): time series data.
        population_cdf (str, or None): The name of a distribution.
        population_args (tuple): Distribution parameter.
        population_loc (float, or None): location of the distribution.
        population_scale (float, or None): scale of the distribution.
        significance_level (float, optional): Significance level. A
            probability threshold below which the null hypothesis will be
            rejected. (default: 0.05)

    Returns:
        bool: True
            Returns True if observed samples are drawn from the same
            continuous distribution as the given distribution (If the
            two-tailed p-value is greater than the significance_level).

    """
    if population_cdf in ('default', None):
        return True

    time_series_data = np.array(time_series_data, copy=False)

    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    cvg_check(significance_level,
              var_name='significance_level',
              var_lower_bound=np.finfo(np.float64).resolution)

    check_population_cdf_args(population_cdf, population_args)

    args = [arg for arg in population_args]
    args.append(population_loc if population_loc else 0)
    args.append(population_scale if population_scale else 1)

    try:
        _, pvalue = kstest(time_series_data,
                           cdf=population_cdf,
                           args=args,
                           alternative='two-sided')
    except:
        raise CVGError('Kolmogorov-Smirnov test failed.')

    return significance_level < pvalue


def levene_test(time_series_data: np.ndarray,
                population_cdf: str,
                population_args: tuple,
                population_loc: float,
                population_scale: float,
                significance_level=1 - _DEFAULT_CONFIDENCE_COEFFICIENT) -> bool:
    """Perform modified Levene test for equal variances.

    The modified Levene test tests the null hypothesis that one sample input
    `time_series_data` is from population `population_cdf` with the same
    variance.

    Note:
        This test is fixed to use 'median' variation of the Levene's test.

        Although the optimal choice depends on the underlying distribution, the
        definition based on the median is recommended as the choice that
        provides good robustness against many types of non-normal data while
        retaining good power.

        Robustness means the ability of the test to not falsely detect unequal
        variances when the underlying data are not normally distributed and the
        variables are in fact equal.

        Power means the ability of the test to detect unequal variances when
        the variances are in fact unequal.

    Args:
        time_series_data (np.ndarray): time series data.
        population_cdf (str, or None): The name of a distribution.
        population_args (tuple): Distribution parameter.
        population_loc (float, or None): location of the distribution.
        population_scale (float, or None): scale of the distribution.
        significance_level (float, optional): Significance level. A
            probability threshold below which the null hypothesis will be
            rejected. (default: 0.05)

    Returns:
        bool: True
            Returns True if the input variance is the same as population
            variance. (If the two-tailed p-value is greater than the
            significance_level).

    References:
        .. [23] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm

    Examples:

    >>> import numpy as np
    >>> from scipy.stats import gamma, alpha
    >>> rng = np.random.RandomState(12345)
    >>> shape, scale = 2., 2.
    >>> x = rng.gamma(shape, scale, size=1000)
    >>> levene_test(x,
                    population_cdf='gamma',
                    population_args=(shape,),
                    population_loc=0,
                    population_scale=scale,
                    significance_level=0.05)
    True

    >>> a = 1.99
    >>> x = gamma.rvs(a, size=1000, random_state=rng)
    >>> levene_test(x,
                    population_cdf='gamma',
                    population_args=(a,),
                    population_loc=0,
                    population_scale=1,
                    significance_level=0.05)
    True

    >>> x = alpha.rvs(a, size=1000, random_state=rng)
    >>> levene_test(x,
                    population_cdf='gamma',
                    population_args=(a,),
                    population_loc=0,
                    population_scale=1,
                    significance_level=0.05)
    False

    Reject the null hypothesis at a confidence level of 5%, concluding that
    there is a difference in variance of the `time_series_data` and `gamma`
    distribution with shape parameter `a`.

    >>> levene_test(x,
                    population_cdf='alpha',
                    population_args=(a,),
                    population_loc=0,
                    population_scale=1,
                    significance_level=0.05)
    True

    """
    if population_cdf in ('default', None):
        return False

    x = np.array(time_series_data, copy=False)

    if x.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    x_size = x.size

    cvg_check(significance_level,
              var_name='significance_level',
              var_lower_bound=np.finfo(np.float64).resolution)

    # population
    check_population_cdf_args(population_cdf, population_args)

    args = [arg for arg in population_args]
    args.append(population_loc if population_loc else 0)
    args.append(population_scale if population_scale else 1)

    rvs = getattr(distributions, population_cdf).rvs

    pvalue = 0.0
    while significance_level > pvalue:
        y = rvs(*args, size=x_size)

        try:
            _, pvalue = kstest(y,
                               cdf=population_cdf,
                               args=args,
                               alternative='two-sided')
        except:
            raise CVGError('Kolmogorov-Smirnov test failed.')

    try:
        _, pvalue = levene(x, y)
    except:
        raise CVGError('Levene test failed.')

    return significance_level < pvalue


def wilcoxon_test(
        time_series_data: np.ndarray,
        population_cdf: str,
        population_args: tuple,
        population_loc: float,
        population_scale: float,
        significance_level=1 - _DEFAULT_CONFIDENCE_COEFFICIENT) -> bool:
    """Calculate the Wilcoxon signed-rank test.

    Here it is used as a non-parametric test to determine whether an unknown
    population mean is different from a specific value.

    Args:
        time_series_data (np.ndarray): time series data.
        population_cdf (str, or None): The name of a distribution.
        population_args (tuple): Distribution parameter.
        population_loc (float, or None): location of the distribution.
        population_scale (float, or None): scale of the distribution.
        significance_level (float, optional): Significance level. A
            probability threshold below which the null hypothesis will be
            rejected. (default: 0.05)

    Returns:
        bool: True
            Returns True if the input sample is the same as population
            distribution.

    Examples:

    >>> import numpy as np
    >>> from scipy.stats import gamma
    >>> rng = np.random.RandomState(12345)
    >>> shape, scale = 2., 2.
    >>> x = rng.gamma(shape, scale, size=1000)
    >>> wilcoxon_test(x,
                      population_cdf='gamma',
                      population_args=(shape,),
                      population_loc=0,
                      population_scale=scale,
                      significance_level=0.05)
    True
    >>> wilcoxon_test(x,
                      population_cdf='gamma',
                      population_args=(shape,),
                      population_loc=0,
                      population_scale=1,
                      significance_level=0.05)
    False

    """
    if population_cdf in ('default', None):
        return False

    x = np.array(time_series_data, copy=False)

    if x.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    x_size = x.size

    cvg_check(significance_level,
              var_name='significance_level',
              var_lower_bound=np.finfo(np.float64).resolution)

    args = [arg for arg in population_args]
    args.append(population_loc if population_loc else 0)
    args.append(population_scale if population_scale else 1)

    rvs = getattr(distributions, population_cdf).rvs

    pvalue = 0.0
    while significance_level > pvalue:
        y = rvs(*args, size=x_size)

        try:
            _, pvalue = kstest(y,
                               cdf=population_cdf,
                               args=args,
                               alternative='two-sided')
        except:
            raise CVGError('Kolmogorov-Smirnov test failed.')

    _, pvalue = wilcoxon(x, y,
                         zero_method='wilcox',
                         alternative='two-sided')

    return significance_level < pvalue


def kruskal_test(
        time_series_data: np.ndarray,
        population_cdf: str,
        population_args: tuple,
        population_loc: float,
        population_scale: float,
        significance_level=1 - _DEFAULT_CONFIDENCE_COEFFICIENT) -> bool:
    """Kruskal-Wallis H-test for independent samples.

    The Kruskal-Wallis H-test tests the null hypothesis that the median of
    the time series data is the same as the one from population_cdf.

    It is a non-parametric version of ANOVA.

    Args:
        time_series_data (np.ndarray): time series data.
        population_cdf (str, or None): The name of a distribution.
        population_args (tuple): Distribution parameter.
        population_loc (float, or None): location of the distribution.
        population_scale (float, or None): scale of the distribution.
        significance_level (float, optional): Significance level. A
            probability threshold below which the null hypothesis will be
            rejected. (default: 0.05)

    Returns:
        bool: True
            if the median of the time series data is the same as the one
            from population_cdf.

    Examples:

    >>> import numpy as np
    >>> from scipy.stats import gamma
    >>> rng = np.random.RandomState(12345)
    >>> a = 1.99
    >>> x = rng.gamma(a, 1, size=20)
    >>> kruskal_test(x,
                     population_cdf='gamma',
                     population_args=(shape,),
                     population_loc=0,
                     population_scale=1,
                     significance_level=0.05)
    True

    """
    if population_cdf in ('default', None):
        return False

    x = np.array(time_series_data, copy=False)

    if x.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    x_size = x.size

    # Due to the assumption that H has a chi square distribution, the number
    # of samples must not be too small. A typical rule is that time_series_data
    # must have at least 5 data.
    if x_size < 5:
        msg = 'time_series_data must have at least 5 data.'
        raise CVGSampleSizeError(msg)

    cvg_check(significance_level,
              var_name='significance_level',
              var_lower_bound=np.finfo(np.float64).resolution)

    args = [arg for arg in population_args]
    args.append(population_loc if population_loc else 0)
    args.append(population_scale if population_scale else 1)

    rvs = getattr(distributions, population_cdf).rvs

    pvalue = 0.0
    while significance_level > pvalue:
        y = rvs(*args, size=x_size)

        try:
            _, pvalue = kstest(y,
                               cdf=population_cdf,
                               args=args,
                               alternative='two-sided')
        except:
            raise CVGError('Kolmogorov-Smirnov test failed.')

    try:
        _, pvalue = kruskal(x, y)
    except:
        raise CVGError('Levene test failed.')

    return significance_level < pvalue
