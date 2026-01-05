Other Utilities
===============

Support tools for data preprocessing, error handling, and quality control.

Batch Means
-----------

Compute batch means from time series data. The ``batch`` function:

* Divides data into non-overlapping batches of size ``batch_size``
* Applies a reduction function (default: ``np.mean``) to each batch
* Supports optional scaling (centering, standardization) **after** batching
* Truncates remainder data points that don't fit into complete batches
* Returns a view (not a copy) when no scaling is requested

Scaling Methods
---------------

Five scaling methods are provided:

* ``minmax_scale`` – Scale to a specified feature range (default: [0, 1])
* ``translate_scale`` – Translate so first element is zero, scale by mean
* ``standard_scale`` – Remove mean and scale to unit variance
* ``robust_scale`` – Center to median, scale by interquartile range
* ``maxabs_scale`` – Scale to [-1, 1] range by maximum absolute value

Each method is available as a function and a class with ``scale()``/``inverse()`` methods.

Error Handling
--------------

Custom exception classes and validation utilities:

* ``CRError`` – Base exception with caller identification
* ``CRSampleSizeError`` – Raised for insufficient data samples
* ``cr_warning()`` – Print warning messages with caller context
* ``cr_check()`` – Validate variable types and bounds
* Decorators: ``_check_ndim``, ``_check_isfinite`` for input validation

Outlier Detection
-----------------

Seven methods to identify outliers:

* ``iqr`` / ``boxplot`` – Points beyond 1.5 × IQR from quartiles
* ``extreme_iqr`` / ``extreme_boxplot`` – Points beyond 3 × IQR
* ``z_score`` / ``standard_score`` – :math:`|Z|` > 3 from mean and std
* ``modified_z_score`` – Robust version using median and MAD (:math:`|Z|` > 3.5)

Returns a 1-D NumPy array of indices or ``None`` if no outliers are found.

Data Splitting
--------------

``train_test_split`` randomly partitions indices for training and testing:

* Supports absolute counts or fractions for train/test sizes
* Validates that splits are feasible given data size
* Accepts an optional ``seed`` for reproducible splits
* Uses NumPy's random number generator internally
* Returns two NumPy index arrays: ``(train_idx, test_idx)``

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from kim_convergence import batch, standard_scale, outlier_test, train_test_split

   data = np.random.randn(1000)

   # Batch the data
   batched = batch(data, batch_size=10)

   # Scale to zero mean, unit variance
   scaled = standard_scale(batched)

   # Check for outliers
   outliers = outlier_test(scaled, outlier_method='iqr')

   # Split for validation
   train_idx, test_idx = train_test_split(data, test_size=0.2, seed=42)

Usage Hints
-----------

* **Batch means**: Use for variance estimation in correlated data
* **Scaling**: Apply ``robust_scale`` when outliers are present
* **Outlier detection**: ``modified_z_score`` works best for small datasets
* **Data splitting**: Set ``seed`` for reproducible cross-validation
