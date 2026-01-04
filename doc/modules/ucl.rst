Upper Confidence Limit (UCL) Module
===================================

Overview
--------

Upper Confidence Limit (UCL): The upper boundary (or limit) of a confidence
interval of a parameter of interest such as the population mean.

A confidence interval is how much uncertainty there is with any particular
statistic [nistdiv898]_.

The UCL module provides statistical methods for estimating confidence intervals,
upper confidence limits, and uncertainty quantification for time series data.
These methods are essential for convergence analysis, simulation output analysis,
and Markov chain Monte Carlo (MCMC) diagnostics.

Confidence limits for the mean are interval estimates. Interval estimates are
often desirable because instead of a single estimate for the mean, a confidence
interval generates a lower and upper limit. It indicates how much uncertainty
there is in our estimation of the true mean. The narrower the gap, the more
precise our estimate is. We use a confidence level to express confidence limits.
Choosing the confidence level is somewhat arbitrary, but 90 %, 95 %, and 99 %
intervals are standard, and 95 % is the most commonly used.

Note:
    One should note that a 95 % confidence interval does not mean a 95 %
    probability of containing the true mean. The interval computed from a
    sample either has the true mean, or it does not. The confidence level is
    simply the proportion of samples of a given size that may be expected to
    contain the true mean. For a 95 % confidence interval, if many samples are
    collected and the confidence interval computed, in the long run, about 95 %
    of these intervals would contain the true mean.


Contents
--------

1. :ref:`Available Methods <ucl-methods>`
2. :ref:`Base Class <ucl-base-class>`
3. :ref:`MSER-m Algorithm <ucl-mser-m>`
4. :ref:`MSER-m-y Algorithm <ucl-mser-m-y>`
5. :ref:`N-SKART Algorithm <ucl-n-skart>`
6. :ref:`Heidelberger-Welch Algorithm <ucl-heidelberger-welch>`
7. :ref:`Uncorrelated Samples Method <ucl-uncorrelated-samples>`
8. :ref:`Usage Examples <ucl-usage-examples>`
9. :ref:`Algorithm Selection Guide <ucl-selection-guide>`
10. :ref:`Performance Considerations <ucl-performance>`
11. :ref:`Theoretical Background <ucl-theory>`

.. _ucl-methods:

Available Methods
-----------------

+---------------------------+--------------------------------------+-------------------------------------------+
| Method                    | Best For                             | Key Features                              |
+===========================+======================================+===========================================+
| **MSER-m**                | Stationary time series               | Batch means, truncation detection         |
+---------------------------+--------------------------------------+-------------------------------------------+
| **MSER-m-y**              | Time series with initial transient   | Combines MSER-m with randomness testing   |
+---------------------------+--------------------------------------+-------------------------------------------+
| **N-SKART**               | Non-normal, correlated data          | Handles skewness and autocorrelation      |
+---------------------------+--------------------------------------+-------------------------------------------+
| **Heidelberger-Welch**    | Spectral analysis approach           | Spectral density at zero frequency        |
+---------------------------+--------------------------------------+-------------------------------------------+
| **Uncorrelated Samples**  | Direct statistical estimation        | Simple t-distribution based CI            |
+---------------------------+--------------------------------------+-------------------------------------------+

.. _ucl-base-class:

Base Class
----------

All UCL methods inherit from ``UCLBase``, which provides a consistent interface
for computing confidence limits.

.. _ucl-mser-m:

MSER-m Algorithm
----------------

.. container:: mser-summary

   .. automodule:: kim_convergence.ucl.mser_m.mser_m
      :members:
      :no-index:

.. raw:: html

   <style>
      /* Hide the 'Args' and 'Returns' fields only inside the summary container */
      .mser-summary .field-list {
          display: none !important;
      }
      /* Optional: Hide the function signature line if you only want the text */
      .mser-summary dt {
          display: none;
      }
   </style>


.. _ucl-mser-m-y:

MSER-m-y Algorithm
------------------

.. container:: mser-y-summary

   .. automodule:: kim_convergence.ucl.mser_m_y.MSER_m_y
      :members:
      :no-index:

.. raw:: html

   <style>
      /* 1. Hide the table-style field lists (standard) */
      .mser-y-summary .field-list {
          display: none !important;
      }

      /* 2. Target specific 'Attributes' or 'Parameters' headers and their content */
      .mser-y-summary dl.field-list,
      .mser-y-summary dl.attribute,
      .mser-y-summary .rubric {
          display: none !important;
      }

      /* 3. Hide the specific 'Attributes' section header generated by Napoleon */
      .mser-y-summary dt:contains('Attributes'),
      .mser-y-summary p.rubric {
          display: none !important;
      }
   </style>

Algorithm Flow
~~~~~~~~~~~~~~

1. Compute k batch means of size m
2. Apply :ref:`MSER-m<ucl-mser-m>` to detect truncation point
3. Apply randomness test to find new batch size :math:`m^*`
4. Iteratively increase batch size until batch means are independent

Key Features
~~~~~~~~~~~~~~

- Automatic detection of independent batch size
- Significance level (probability threshold below which the null hypothesis
  will be rejected) of 0.2 for randomness test
- Minimum data requirement: 10 points

.. _ucl-n-skart:

N-SKART Algorithm
-----------------

.. container:: n-skart

   .. automodule:: kim_convergence.ucl.n_skart.N_SKART
      :members:
      :no-index:

.. raw:: html

   <style>
      /* 1. Hide the table-style field lists (standard) */
      .n-skart .field-list {
          display: none !important;
      }

      /* 2. Target specific 'Attributes' or 'Parameters' headers and their content */
      .n-skart dl.field-list,
      .n-skart dl.attribute,
      .n-skart .rubric {
          display: none !important;
      }

      /* 3. Hide the specific 'Attributes' section header generated by Napoleon */
      .n-skart dt:contains('Attributes'),
      .n-skart p.rubric {
          display: none !important;
      }
   </style>

Algorithm Highlights
~~~~~~~~~~~~~~~~~~~~

- Adjusts for skewness using sample skewness of last 80% of data
- Uses von Neumann randomness test with spacer batches
- Includes correlation and skewness adjustments in UCL calculation

.. _ucl-heidelberger-welch:

Heidelberger-Welch Algorithm
----------------------------

.. container:: heidelberger-welch

   .. automodule:: kim_convergence.ucl.spectral.HeidelbergerWelch._ucl_impl
      :members:
      :no-index:

.. raw:: html

   <style>
      /* 1. Hide the table-style field lists (standard) */
      .heidelberger-welch .field-list {
          display: none !important;
      }

      /* 2. Target specific 'Attributes' or 'Parameters' headers and their content */
      .heidelberger-welch dl.field-list,
      .heidelberger-welch dl.attribute,
      .heidelberger-welch .rubric {
          display: none !important;
      }

      /* 3. Hide the specific 'Attributes' section header generated by Napoleon */
      .heidelberger-welch dt:contains('Attributes'),
      .heidelberger-welch p.rubric {
          display: none !important;
      }
   </style>

Key Features
~~~~~~~~~~~~

- Adaptive polynomial fitting (degrees 1-3) to periodogram
- Uses modified periodogram for spectral estimation
- Optimal for stationary time series with unknown correlation structure

.. _ucl-uncorrelated-samples:

Uncorrelated Samples Method
---------------------------

Direct method that computes confidence intervals using uncorrelated subsamples
of the time series, with optional population standard deviation.

Two Modes of Operation
~~~~~~~~~~~~~~~~~~~~~~

1. **Population standard deviation known:**

   .. math::

      UCL = t_{\alpha,d} \left(\frac{\sigma}{\sqrt{n}}\right)

2. **Population standard deviation unknown:**

   .. math::

      UCL = t_{\alpha,d} \left(\frac{s}{\sqrt{n}}\right)

where :math:`\sigma` is population standard deviation, :math:`s` is sample
standard deviation, and :math:`t_{\alpha,d}` is the t-distribution critical
value. This value depends on the `confidence_coefficient` and the degrees of
freedom, which is found by subtracting one from the number of observations.

Sampling Methods
~~~~~~~~~~~~~~~~~~~~~~

- ``uncorrelated``: Systematic sampling based on statistical inefficiency
- ``random``: Random sampling
- ``block_averaged``: Block-averaged sampling

.. _ucl-usage-examples:

Usage Examples
--------------

Basic Usage Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kim_convergence.ucl import MSER_m
   import numpy as np

   # Generate sample data
   np.random.seed(42)
   data = np.random.normal(loc=10, scale=2, size=5000)

   # Initialize algorithm
   mser = MSER_m()

   # Estimate equilibration length
   truncated, truncation_point = mser.estimate_equilibration_length(data)

   # Compute upper confidence limit
   ucl_value = mser.ucl(data, confidence_coefficient=0.95)

   # Get confidence interval
   lower, upper = mser.ci(data, confidence_coefficient=0.95)

   # Get relative half-width
   relative_width = mser.relative_half_width_estimate(data, confidence_coefficient=0.95)

Comparing Different Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from kim_convergence.ucl import (
       mser_m_ucl, mser_m_y_ucl, n_skart_ucl,
       heidelberger_welch_ucl, uncorrelated_samples_ucl
   )

   # Generate autocorrelated data
   np.random.seed(42)
   n = 5000
   phi = 0.7
   data = np.zeros(n)
   data[0] = np.random.normal(loc=10, scale=2)
   for i in range(1, n):
       data[i] = phi * data[i-1] + np.random.normal(loc=10*(1-phi), scale=2*np.sqrt(1-phi**2))

   # Compare UCL values
   results = {
       'MSER-m': mser_m_ucl(data, confidence_coefficient=0.95),
       'MSER-m-y': mser_m_y_ucl(data, confidence_coefficient=0.95),
       'N-SKART': n_skart_ucl(data, confidence_coefficient=0.95),
       'Heidelberger-Welch': heidelberger_welch_ucl(data, confidence_coefficient=0.95),
       'Uncorrelated Samples': uncorrelated_samples_ucl(data, confidence_coefficient=0.95)
   }

Handling Skewed and Correlated Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kim_convergence.ucl import N_SKART
   import numpy as np

   # Create skewed, correlated data
   np.random.seed(42)
   n = 3000
   base_data = np.random.gamma(shape=2.0, scale=1.0, size=n)

   # Add autocorrelation
   correlated_data = np.zeros(n)
   correlated_data[0] = base_data[0]
   phi = 0.6
   for i in range(1, n):
       correlated_data[i] = phi * correlated_data[i-1] + (1-phi) * base_data[i]

   # Use N-SKART
   nskart = N_SKART()
   truncated, eq_point = nskart.estimate_equilibration_length(correlated_data)
   ucl = nskart.ucl(correlated_data, confidence_coefficient=0.95)


.. _ucl-selection-guide:

Algorithm Selection Guide
-------------------------

Choosing the Right Method
~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+----------------+----------------+----------------+----------------+----------------------+
| Consideration        | MSER-m         | MSER-m-y       | N-SKART        | Heidelberger-  | Uncorrelated         |
|                      |                |                |                | Welch          | Samples              |
+======================+================+================+================+================+======================+
| **Min Data Points**  | > batch_size   | 10             | 1280           | 200            | 5                    |
+----------------------+----------------+----------------+----------------+----------------+----------------------+
| **Autocorrelation**  | Moderate       | High           | **High**       | **High**       | Low                  |
+----------------------+----------------+----------------+----------------+----------------+----------------------+
| **Skewness**         | Not robust     | Not robust     | **Robust**     | Not robust     | Not robust           |
+----------------------+----------------+----------------+----------------+----------------+----------------------+
| **Initial Transient**| Manual removal | **Automatic**  | **Automatic**  | Manual removal | Manual removal       |
+----------------------+----------------+----------------+----------------+----------------+----------------------+
| **Speed**            | **Fast**       | Fast           | Moderate       | Fast           | **Fast**             |
+----------------------+----------------+----------------+----------------+----------------+----------------------+

Quick Decision Tree
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Is your data approximately normal and stationary?**
   - Yes -> Use **MSER-m** or **Uncorrelated Samples**
   - No -> Go to 2

2. **Does your data have significant initial transient (burn-in)?**
   - Yes -> Use **MSER-m-y** or **N-SKART**
   - No -> Go to 3

3. **Is your data highly skewed or heavy-tailed?**
   - Yes -> Use **N-SKART**
   - No -> Use **Heidelberger-Welch**


.. _ucl-performance:

Performance Considerations
--------------------------

Memory Usage
~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+----------------+------------------------------------------------+
| Method               | Memory Usage   | Notes                                          |
+======================+================+================================================+
| MSER-m               | O(N)           | Stores original data and batch means           |
+----------------------+----------------+------------------------------------------------+
| MSER-m-y             | O(N)           | Similar to MSER-m with additional test arrays  |
+----------------------+----------------+------------------------------------------------+
| N-SKART              | O(N + k)       | Stores data and multiple batch statistics      |
+----------------------+----------------+------------------------------------------------+
| Heidelberger-Welch   | O(N + k)       | Stores data and periodogram (k=50 default)     |
+----------------------+----------------+------------------------------------------------+
| Uncorrelated Samples | O(N)           | Stores data and uncorrelated sample indices    |
+----------------------+----------------+------------------------------------------------+

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+----------------+------------------------------------------------+
| Method               | Complexity     | Notes                                          |
+======================+================+================================================+
| MSER-m               | O(N)           | Single pass through data                       |
+----------------------+----------------+------------------------------------------------+
| MSER-m-y             | O(N log N)     | Multiple iterations with increasing batch size |
+----------------------+----------------+------------------------------------------------+
| N-SKART              | O(N log N)     | Iterative batch size adjustment and testing    |
+----------------------+----------------+------------------------------------------------+
| Heidelberger-Welch   | O(N log N)     | FFT-based periodogram computation              |
+----------------------+----------------+------------------------------------------------+
| Uncorrelated Samples | O(N)           | Single pass with optional FFT for SI           |
+----------------------+----------------+------------------------------------------------+

Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~

**Data Preparation**

1. Ensure data is 1-dimensional numpy array or list
2. Remove obvious outliers using ``kim_convergence.outlier`` module
3. Check stationarity visually or with statistical tests
4. Consider differencing for non-stationary time series

**Parameter Tuning**

1. **MSER-m/ MSER-m-y**: Set ``batch_size`` ≈ 5-10 times the expected correlation time
2. **N-SKART**: Let algorithm determine parameters automatically
3. **Heidelberger-Welch**: Increase ``heidel_welch_number_points`` for smoother spectra
4. **Uncorrelated Samples**: Provide ``population_standard_deviation`` if known

**Validation and Diagnostics**

1. Always check the ``truncated`` flag from ``estimate_equilibration_length()``
2. Compare results from 2-3 different methods for consistency
3. Use ``relative_half_width_estimate()`` to assess precision

Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~

All functions raise appropriate exceptions:

- ``CRError``: For general errors and invalid inputs
- ``CRSampleSizeError``: For insufficient sample sizes
- Value errors for out-of-range parameters (e.g., confidence coefficient ∉ (0,1))


.. _ucl-theory:

Theoretical Background
----------------------

Batch Means Methods (MSER-m, MSER-m-y, N-SKART)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch means methods operate on the principle that for a stationary time series,
batch means become approximately independent and normally distributed as batch
size increases. The variance of the overall mean can be estimated from the
variance of batch means:

.. math::

   \text{Var}(\bar{X}) \approx \frac{s_b^2}{k}

where :math:`s_b^2` is the sample variance of batch means and :math:`k` is the
number of batches.

Spectral Methods (Heidelberger-Welch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spectral methods estimate the variance of the mean via the spectral density
function. For a covariance stationary process:

.. math::

   \text{Var}(\bar{X}_N) = \frac{1}{N} \sum_{k=-(N-1)}^{N-1} \left(1 - \frac{|k|}{N}\right) \gamma(k)

where :math:`\gamma(k)` is the autocovariance at lag k.

Direct Methods (Uncorrelated Samples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct methods use the classical formula for the variance of the mean,
adjusted for effective sample size when data is correlated:

.. math::

   \text{Var}(\bar{X}) = \frac{s^2}{N_{\text{eff}}} = \frac{s^2}{N / g}

where :math:`g` is the statistical inefficiency and :math:`N_{\text{eff}}` is
the effective sample size.
