Time Series Analysis Module
===========================

Analyze correlated time-series data from simulations, experiments, or
any sequential measurement process.

Capabilities
------------

* Automatic equilibration detection
* Integrated autocorrelation time / statistical inefficiency estimation
* Geweke convergence diagnostic
* Sub-sampling utilities: regular, random, or block-averaged

Equilibration Length Estimation
-------------------------------

Scan candidate truncation points and return the one that maximizes the
effective sample size using the automated procedure by Chodera
[chodera2016]_. The search grid can be thinned with ``nskip`` and
the tail of the data can be ignored via ``ignore_end`` (int points, float
fraction, or None for one-quarter). Geyer estimators impose larger
minimum tail lengths. Parallel evaluation is supported with
``number_of_cores``. Constant series are detected automatically and
return ``(0, n_data)``.

Statistical Inefficiency
------------------------

Four estimators are provided:

* ``statistical_inefficiency`` – standard integrated autocorrelation time
* ``geyer_r_statistical_inefficiency`` – Geyer initial monotone sequence
  [geyer1992]_, [geyer2011]_
* ``geyer_split_r_statistical_inefficiency`` – split-chain variant (needs ≥ 8 points)
* ``geyer_split_statistical_inefficiency`` – split-chain minimum variance

All functions accept an optional second array ``y`` to compute
cross-correlation instead of auto-correlation. FFT convolution is used
by default for series longer than 30 points; it can be disabled with
``fft=False``. The summation window is controlled with
``minimum_correlation_time``. Constant data return ``si = n_data``.

Geweke Diagnostic
-----------------

Compute z-scores by comparing early and late segments of the chain
[geweke1992]_, [plummer2006]_. Signature::

   geweke(x, first=0.1, last=0.5, intervals=20)

Returns an (intervals, 2) array whose columns are [start index, z-score].

Sampling Utilities
------------------

Three modes are available through a single entry point:

* ``uncorrelated`` – take every ``si``-th point
* ``random`` – one random point per ``si``-length block
* ``block_averaged`` – average of each ``si``-length block

Pre-computed indices can be supplied via ``uncorrelated_sample_indices``
to avoid recomputing the statistical inefficiency.

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from kim_convergence.timeseries import (
       estimate_equilibration_length,
       statistical_inefficiency,
       geweke,
       uncorrelated_time_series_data_samples
   )

   data = np.random.randn(10000)          # your correlated series

   eq, si = estimate_equilibration_length(data, nskip=10)
   z      = geweke(data[eq:], intervals=20)
   uncorr = uncorrelated_time_series_data_samples(
                data[eq:], si=si, sample_method='block_averaged')

Hints
-----

* Geyer estimators are preferred for noisy or slowly-decaying correlations
* Block averaging is best for thermodynamic observables
* Use ``ignore_end`` to exclude non-stationary tails
* Set ``number_of_cores > 1`` to accelerate long scans
* For series shorter than 30 points, FFT is disabled automatically
* The module works with any time-ordered data, not just simulation trajectories
