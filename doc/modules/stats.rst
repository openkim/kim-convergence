Statistics Module
=================

Overview
--------

The statistics module provides tools for statistical analysis, hypothesis
testing, and time series analysis. It's designed for use in convergence analysis
and statistical quality control applications.

Contents
--------

.. toctree::
   :maxdepth: 2

1. :ref:`Distribution Functions <stats-distributions>`
2. :ref:`Hypothesis Tests <stats-hypothesis-tests>`
3. :ref:`Time Series Analysis <stats-time-series>`
4. :ref:`Randomness Tests <stats-randomness>`
5. :ref:`Common Usage Patterns <stats-usage-patterns>`
6. :ref:`Performance Considerations <stats-performance>`

.. _stats-distributions:

Distribution Functions
----------------------

Beta Distribution
~~~~~~~~~~~~~~~~~

.. automodule:: kim_convergence.stats.beta_dist
   :members:
   :exclude-members: __all__

The Beta function implementation follows the algorithms described in
Numerical Recipes [numrec2007]_.

Normal Distribution
~~~~~~~~~~~~~~~~~~~

.. automodule:: kim_convergence.stats.normal_dist
   :members:
   :exclude-members: __all__

The inverse CDF computation uses the algorithm by Wichura [wichura1988]_.

t-Distribution
~~~~~~~~~~~~~~

.. automodule:: kim_convergence.stats.t_dist
   :members:
   :exclude-members: __all__

The t-distribution functions are implemented using the regularized
incomplete beta function as described in standard statistical references.

.. _stats-hypothesis-tests:

Hypothesis Tests
----------------

Tests for Normally Distributed Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: kim_convergence.stats.normal_test
   :members:
   :exclude-members: __all__

Tests for Non-Normally Distributed Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: kim_convergence.stats.nonnormal_test
   :members: ks_test, levene_test, wilcoxon_test, kruskal_test

The non-parametric tests in this module rely on distributions from SciPy
[scipystats]_. Available non-parametric tests include:

- **Kolmogorov-Smirnov test** (``ks_test``): Tests if samples come from a given
  distribution
- **Levene's test** (``levene_test``): Tests for equal variances [nistdiv898b]_
- **Wilcoxon signed-rank test** (``wilcoxon_test``): Tests if median differs
  from a value
- **Kruskal-Wallis H-test** (``kruskal_test``): Non-parametric version of ANOVA

.. _stats-time-series:

Time Series Analysis Tools
--------------------------

.. automodule:: kim_convergence.stats.tools
   :members:
   :exclude-members: __all__, FFTURN, get_fft_optimal_size

.. _stats-randomness:

Randomness Test
---------------

.. automodule:: kim_convergence.stats.randomness_test
   :members:
   :no-index:
   :exclude-members: __all__

.. _stats-usage-patterns:

Common Usage Patterns
---------------------

Testing if data is normally distributed:

.. code-block:: python

    import numpy as np
    from kim_convergence.stats import t_test

    # Generate sample data
    data = np.random.normal(loc=0, scale=1, size=100)

    # Perform t-test against population mean of 0
    result = t_test(
        sample_mean=np.mean(data),
        sample_std=np.std(data),
        sample_size=len(data),
        population_mean=0,
        significance_level=0.05
    )

    print(f"Null hypothesis accepted: {result}")

Checking time series randomness:

.. code-block:: python

    from kim_convergence.stats.randomness_test import randomness_test

    # Check if time series exhibits independence
    is_random = randomness_test(time_series_data, significance_level=0.05)

    if is_random:
        print("Time series appears independent")
    else:
        print("Time series shows serial correlation")

Testing against a specific distribution:

.. code-block:: python

    from kim_convergence.stats.nonnormal_test import ks_test

    # Test if data comes from a gamma distribution
    is_gamma = ks_test(
        time_series_data,
        population_cdf='gamma',
        population_args=(2.0,),  # shape parameter
        population_loc=0,
        population_scale=1.0,
        significance_level=0.05
    )

Computing autocorrelation:

.. code-block:: python

    from kim_convergence.stats.tools import auto_correlate

    # Compute autocorrelation with FFT optimization
    autocorr = auto_correlate(time_series_data, nlags=50, fft=True)

    # First few lags (excluding lag 0 which is always 1.0)
    print(f"Autocorrelation at lag 1: {autocorr[1]:.3f}")
    print(f"Autocorrelation at lag 2: {autocorr[2]:.3f}")

.. _stats-performance:

Performance Considerations
--------------------------

FFT Optimization
~~~~~~~~~~~~~~~~

For long time series, always use ``fft=True`` in autocorrelation functions:

.. code-block:: python

    # For time series with > 1000 points
    autocorr = auto_correlate(large_time_series, fft=True)
    crosscorr = cross_correlate(x, y, fft=True)

The ``get_fft_optimal_size()`` function finds optimal sizes for FFT computations
by returning the smallest 5-smooth number (factors 2, 3, 5 only) greater than
or equal to the input size [statsmodels]_.

Sample Size Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

- **Non-parametric tests**: Require at least 5 data points
- **Randomness test**: Requires at least 3 data points
- **t-distribution functions**: Degrees of freedom must be > 1
- **KS test**: Most effective with moderate to large sample sizes (>30)

Numerical Stability
~~~~~~~~~~~~~~~~~~~

- Use ``bias=False`` in ``skew()`` function for unbiased estimation
- Distribution functions handle edge cases (e.g., p=0, p=1) appropriately

Memory Usage
~~~~~~~~~~~~

- FFT-based functions create temporary arrays of optimal FFT size
- Auto/cross-covariance functions return arrays of length N (not 2N-1)
- Consider using ``nlags`` parameter to limit output size for long series

Error Handling
~~~~~~~~~~~~~~

All functions raise appropriate exceptions:

- ``CRError``: For general errors and invalid inputs
- ``CRSampleSizeError``: For insufficient sample sizes
- Value errors for out-of-range parameters (e.g., p ∉ [0,1])
