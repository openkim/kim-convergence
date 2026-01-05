Best Practices
==============

Selecting Accuracy Requirements
-------------------------------

**Relative vs. Absolute Accuracy**

* Use **relative accuracy** when you care about percentage precision relative to
  the mean value. Ideal for quantities like densities, concentrations, or
  normalized properties.
* Use **absolute accuracy** when the absolute magnitude matters more than
  relative percentage. Appropriate for temperatures, pressures, or energies
  where fixed error bounds are meaningful.

**Hard Limits**

* Absolute accuracy must be ≥ 1 × 10⁻⁴; smaller values raise CRError.
* Relative accuracy must be > 0 and < 1.

**Guidelines**

- Start with ``relative_accuracy=0.01`` (1%) for general scientific work.
- Switch to ``absolute_accuracy`` when the observable crosses zero or has
  negligible magnitude.
- For multiple observables, set accuracies only on the quantities that matter
  scientifically; the run continues until **every** supplied threshold is met.

Choosing UCL / CI Methods
-------------------------

The package first **truncates** the series (MSER-m) and then **estimates
uncertainty** with one of the methods below.

* **MSER-m**: selects the equilibration point that minimizes the standard error
  of the mean; the UCL is then computed with the chosen CI method.
* **Heidelberger-Welch**: spectral variance estimator; good for complex
  correlations.
* **N-SKART**: non-overlapping batch means with skewness adjustment; useful for
  non-normal data.
* **Uncorrelated_Samples**: classical confidence interval on the
  effective-sample-size corrected mean.

Start with the default method; compare two methods to check consistency.

Monitoring Multiple Observables
-------------------------------

Different quantities converge at different rates.

.. code-block:: python

   result = cr.run_length_control(
       get_trajectory=get_trajectory,
       number_of_variables=3,
       relative_accuracy=[0.01, 0.02, 0.005],
       absolute_accuracy=[None, 0.1, None],
       maximum_run_length=1_000_000,
   )

The simulation stops only when **all** observables satisfy their own
accuracy requirement.

Setting Initial Parameters
--------------------------

**initial_run_length**

* 100–1,000 points are usually enough for simple properties; complex systems
  may need ≥10,000.
* Too small: unreliable equilibration detection; too large: wasted steps.

**maximum_equilibration_step**

* Choose 30–50% of ``maximum_run_length`` to allow slow equilibration
  without hitting the hard run cap.

**minimum_number_of_independent_samples**

* Leave at the default unless you need a specific effective-sample-size
  guarantee.

Validating Results
------------------

Inspect the returned JSON/EDN report:

1. **equilibration_step** – should be physically reasonable.
2. **statistical_inefficiency** – ≥1; very large values mean strong
   correlations and longer runs.
3. **effective_sample_size** – should exceed the minimum required by your
   later statistical tests (≥30 is a common rule of thumb).
4. Run the same job with two UCL methods; large discrepancies warn that the
   series is still far from stationary.

Performance Optimization
------------------------

* **FFT** – Enabled automatically for n > 30; disable only for short series
  or debugging.
* **Parallel equilibration scan** – Set ``number_of_cores > 1`` in
  ``estimate_equilibration_length`` (does **not** parallelize the main
  convergence loop).
* **nskip** – Increase to stride over candidate truncation points when the
  series is very long.
* **Batch means** – Use ≥20 batches for stable variance estimates.

Common Pitfalls
---------------

1. **Unrealistic accuracy** – 0.1% can require orders of magnitude more data
   than 1% and may simply hit ``maximum_run_length``.
2. **Ignoring correlation** – Assuming independence underestimates uncertainty.
3. **Too small ``maximum_run_length``** – The run finishes with
   ``"converged": false``.
4. **Constant observable** – The code detects this automatically (si = n_data);
   no user action is required.
5. **Memory pressure** – Dump very long trajectories to disk with
   ``dump_trajectory=True``.

Integration with Simulators
---------------------------

* **Callbacks** – Provided for LAMMPS and OpenMM; return an array of shape
  ``(n_variables, n_steps)``.
* **Custom codes** – Implement ``get_trajectory(step)`` or
  ``get_trajectory(step, args)`` and pass the same signature to
  ``run_length_control``.
* **Checkpointing** – Write the convergence report to file periodically;
  the trajectory can be dumped with ``dump_trajectory_fp``.
* **Real-time monitoring** – Stream the JSON report to stdout or a file
  descriptor and parse it on the fly.

When to Override Defaults
-------------------------

Change parameters only when:

* You have prior knowledge of correlation length or convergence speed.
* Domain-specific literature recommends different accuracy levels.
* Exploratory runs show that defaults produce physically implausible
  equilibration points or inconsistent UCL estimates.
* Computational budget forces a trade-off between speed and precision.

Remember: The goal is statistically sound, reproducible results obtained
with the least computational effort.
