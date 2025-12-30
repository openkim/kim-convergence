Theory and Background
=====================

Overview
--------

The package addresses two fundamental problems in simulation analysis:

1. **Equilibration detection**: Identifying when a simulation reaches stationarity
2. **Convergence assessment**: Determining when estimates are sufficiently precise

The workflow first discards non-stationary data, then uses statistical inefficiency
to account for correlations, and finally compares uncertainty estimates against
user-specified accuracy thresholds.

Equilibration Detection
-----------------------

Initial simulation data often exhibits transient behavior. The package implements
the method by Chodera [chodera2016]_ that selects the truncation point maximizing
the effective sample size:

.. math::
   t_{\text{eq}} = \arg\max_t \frac{N - t}{si(t)}

where :math:`si(t)` is the statistical inefficiency computed on data :math:`[t:N]`.

Statistical Inefficiency and Effective Sample Size
--------------------------------------------------

For correlated time series, the statistical inefficiency :math:`si` quantifies
the reduction in independent information:

.. math::
   si = 1 + 2 \sum_{\tau=1}^{N-1} \left(1 - \frac{\tau}{N}\right) \rho(\tau)

where :math:`\rho(\tau)` is the normalized autocorrelation at lag :math:`\tau`.
The effective sample size is then:

.. math::
   N_{\text{eff}} = \frac{N}{si}

Multiple estimators are provided:

* Integrated autocorrelation time (standard estimator)
* Geyer's initial monotone sequence [geyer1992]_, [geyer2011]_
* Split-chain variants for improved variance estimation

Uncertainty Quantification
--------------------------

Upper Confidence Limits (UCLs) estimate the uncertainty in mean values:

* **MSER-m** (Minimum Standard Error Rule): Finds truncation point minimizing
  the standard error of the mean [white1997]_, [spratt1998]_:

  .. math::
     \text{MSER}_m(d) = \frac{1}{(n-d)^2} \sum_{j=d+1}^n (Y_j - \bar{Y}_{d:n})^2

* **Heidelberger-Welch**: Spectral method for variance estimation [heidelberger1981]_
* **N-SKART**: Non-overlapping batch means with skewness adjustment [tafazzoli2011]_
* **Uncorrelated Samples**: Assumes samples are independent after accounting for
  statistical inefficiency

Batch Means Method
------------------

For correlated data, batch means provides consistent variance estimates.
Data is divided into :math:`b` non-overlapping batches of size :math:`m`,
and the variance of batch means estimates the true variance:

.. math::
   \hat{\sigma}^2 = \frac{m}{b-1} \sum_{i=1}^b (\bar{Y}_i - \bar{Y})^2

Hypothesis Testing
------------------

Supporting tests include:

* **t-tests** for mean comparisons
* **Chi-square tests** for variance analysis
* **Levene's test** for homogeneity of variances
* **Von Neumann test** for randomness [vonneumann1941]_

Implementation Details
----------------------

See the module documentation for:

* Algorithm specifics and computational considerations
* Parameter selection guidelines
* Practical usage examples
