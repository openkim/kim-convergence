Troubleshooting
===============

Common Issues
-------------

"Equilibration Not Detected"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: The algorithm fails to identify a stationary region, or returns an
unexpectedly large equilibration index.

**Solutions**:

1. Increase ``maximum_equilibration_step`` to allow more warm-up steps
2. Use ``initial_run_length`` to provide sufficient initial data
3. Consider if your observable is appropriate for equilibration detection
   (some quantities may oscillate without reaching stationarity)
4. Check your simulation setup for convergence issues
5. Use ``ignore_end`` to exclude non-stationary tail behavior

"Failed to Compute UCL"
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Insufficient data for reliable uncertainty quantification, or the
UCL method cannot produce a valid estimate.

**Solutions**:

1. Reduce ``relative_accuracy`` or ``absolute_accuracy`` requirements
2. Increase ``maximum_run_length`` to collect more data
3. Check for constant or near-constant data (see "Edge Cases" below)
4. Verify that ``minimum_sample_size`` requirements are met
5. Try a different UCL method (MSER-m, Heidelberger-Welch, etc.)

"Relative Accuracy Ill-Defined"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: The confidence interval includes zero, making relative accuracy
calculations meaningless.

**Solutions**:

1. Use ``absolute_accuracy`` instead of ``relative_accuracy``

"Performance Issues with Large Datasets"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Long simulations or high-frequency sampling cause slow computation.

**Solutions**:

1. Enable ``fft=True`` (default) for fast correlation calculations
2. Reduce ``minimum_correlation_time`` to truncate autocorrelation sums earlier
3. Use ``number_of_cores`` for parallel processing with multiple variables
4. Increase ``nskip`` in ``estimate_equilibration_length`` for faster scanning
5. Consider subsampling your data if temporal resolution exceeds correlation time

"Statistical Inefficiency Returns Data Size"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ``si`` equals the full data size (worst-case correlation).

**Causes**:

1. Constant or near-constant time series (zero variance)
2. Computational precision issues with very small variance
3. Autocorrelation function fails to decay

**Solutions**:

1. Add small numerical noise to constant data (see "Edge Cases" below)
2. Check data variance exceeds machine precision (``np.std(data) > 1e-15``)
3. Verify your simulation is producing meaningful fluctuations

Edge Cases
----------

Constant or Near-Constant Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many statistical methods require finite variance. For testing or degenerate cases:

.. code-block:: python

   def get_trajectory(nstep):
       data = your_simulation(nstep)
       # Add tiny noise if variance is below machine precision
       if np.std(data) < 1e-10:
           data += np.random.normal(0, 1e-8, nstep)
       return data

Insufficient Sample Size
~~~~~~~~~~~~~~~~~~~~~~~~

**Error**: "not enough data points" or CRSampleSizeError.

**Fix**:

- Ensure ``initial_run_length`` > minimum requirements (varies by method)
- Geyer methods require ≥4 points, split methods require ≥8 points
- Increase data collection before calling analysis functions

Non-Finite Values (NaN/inf)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error**: "non-finite or not-number" detected.

**Fix**: Clean your data in the ``get_trajectory`` function:

.. code-block:: python

   def get_trajectory(nstep):
       data = your_simulation(nstep)
       # Replace NaN/inf with neighboring values or remove
       mask = np.isfinite(data)
       if not np.all(mask):
           # Simple linear interpolation for missing values
           data = np.interp(np.arange(nstep),
                           np.where(mask)[0],
                           data[mask])
       return data

Unexpected Convergence Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If convergence seems too fast or too slow:

1. Check that accuracy requirements are physically meaningful for your observable
2. Verify UCL method assumptions match your data characteristics
3. Use diagnostic tools (Geweke z-scores, autocorrelation plots) to inspect data
4. Compare results across multiple UCL methods for consistency

Deadlock or hang inside simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your simulation (e.g., LAMMPS with OpenMP) deadlocks or severely slows down
when using kim-convergence, set the following environment variable **before**
launching the simulation:

.. code-block:: bash

   export KIM_CONV_FORCE_SUBPROC=1

This forces correlation and FFT computations into isolated subprocesses,
preventing threading conflicts with the host simulator's parallelism.

**Important performance note**:
- In production simulations with large datasets: moderate overhead (typically
  10–30%).
- In unit tests or with small data: **extremely high overhead** (can be 1000x
  slower) due to process spawning costs, especially on macOS.

**Never** set this variable when running tests. It is intended only as an
escape hatch for real simulation runs that exhibit threading deadlocks.

If the problem persists after enabling this flag, please open an issue with
details about your simulation setup and kim-convergence version.

Debugging Tips
--------------

1. **Start simple**: Use default parameters first, then customize
2. **Validate inputs**: Check data shape, dtype, and range before passing
3. **Use diagnostic outputs**: Set ``fp='return'`` to examine detailed statistics
4. **Test with synthetic data**: Create known-cases to verify behavior
5. **Check module-specific documentation**: Each module may have specific requirements

For further assistance, consult the API documentation or module examples.
