Run-Length Control Module
==========================

.. admonition:: 30-second start
   :class: tip

   .. code-block:: python

      import json
      import numpy as np

      from kim_convergence import run_length_control

      def my_sim(n):
         return np.random.normal(0, 1, n)

      result_str = run_length_control(
          get_trajectory=lambda n: my_sim(n),
          relative_accuracy=0.05,          # 5% relative error
          initial_run_length=2_000,
          maximum_run_length=200_000,
          fp="return",
          fp_format="json",
      )
      result = json.loads(result_str)
      print(f"Converged after {result['total_run_length']:,} steps")
      print(f"UCL = {result['upper_confidence_limit']:.3g}")

   Increase ``maximum_run_length`` or relax ``relative_accuracy`` if the
   simulation stops early.

An adaptive algorithm that runs your simulation **only as long as necessary** to
reach user-defined statistical accuracy.  The module mainly follows a two-stage
process:

* **Equilibration detection**: Extends the trajectory using MSER truncation
  until a stationary region is found, then it refines the truncation point with
  integrated autocorrelation time analysis on the existing data
* **Convergence checking**: After equilibration is detected, further extend
  the trajectory until all variables meet statistical accuracy criteria
  (relative or absolute)

Both stages use geometric extension controlled by ``run_length_factor``, but
once equilibration is detected, the algorithm never re-checks for it.
Optional population parameter validation can be enabled to test the results
against known distributions.

Algorithm Flow
--------------

.. mermaid::
   :caption: Run-length control algorithm flow

   flowchart TD
       Start([Start]) --> Setup[Setup & Validation]
       Setup --> Equil[Equilibration Detection<br/>MSER truncation]
       Equil --> Detected{Equilibration<br/>detected by MSER?}
       Detected -->|Yes| Refine[Refine truncation point<br/>with integrated autocorrelation time]
       Detected -.->|No| Extend1[Extend trajectory<br/>x run_length_factor]
       Extend1 -.->|max_equil_step| FailEquil[! Give up: equilibration<br/>not detected]
       Extend1 --> Equil

       Refine --> CheckStep{Equilibration step<br/><= maximum_equilibration_step?}
       CheckStep -->|Yes| Conv[Convergence Stage<br/>UCL check]
       CheckStep -.->|No| FailEquil2[! Give up: equilibration<br/>exceeds hard limit]

       Conv --> AllConverged{All variables<br/>converged?}
       AllConverged -->|Yes| Report[Build report]
       AllConverged -.->|No| MaxLength{Max length<br/>reached?}
       MaxLength -->|Yes| Final[Compute final stats<br/>may be unreliable]
       MaxLength -.->|No| Extend2[Extend trajectory<br/>x run_length_factor]
       Extend2 --> Conv

       Final --> Report
       Report --> Output([Output results])

Core Concepts
-------------

Equilibration
   Time after which the simulation has forgotten its initial condition and
   behaves like the stationary process.

UCL
   Upper confidence limit; half-width of a two-sided confidence interval.

run_length_factor
   Multiplicative factor by which the trajectory is extended when
   convergence has not yet been reached.

Equilibration Detection
~~~~~~~~~~~~~~~~~~~~~~~

1. **MSER-m** minimises the marginal standard error of batch means.
2. Optional **3-sigma** test against known population mean.
3. **Integrated autocorrelation time** refines the truncation point.

Convergence Checking
~~~~~~~~~~~~~~~~~~~~~~~

After discarding the warm-up period, the algorithm repeatedly

* recomputes the **UCL** for every variable,
* compares it with the requested **relative** or **absolute** accuracy,
* extends the trajectory by ``run_length_factor`` until **all** variables pass
  **or** ``maximum_run_length`` is reached.

Parameter Schema
----------------

.. list-table:: Frequently used options
   :widths: 25 50 25
   :header-rows: 1

   * - Parameter
     - Purpose
     - Typical value
   * - ``relative_accuracy``
     - Target relative precision (UCL/ :math:`|mean|`)
     - 0.01 - 0.10
   * - ``absolute_accuracy``
     - Target absolute precision (UCL)
     - 0.1 - 1.0
   * - ``run_length_factor``
     - Growth rate between checkpoints
     - 1.5 - 2.0
   * - ``maximum_run_length``
     - Hard budget (total simulation steps)
     - 1e4 - 1e7
   * - ``initial_run_length``
     - First segment supplied to the algorithm
     - 1e3 - 1e4

Full signature is automatically validated by the internal
model (types, defaults, mutual exclusions).

Usage Examples
--------------

Single Variable, 5% Relative Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = run_length_control(
       get_trajectory=my_simulator,
       relative_accuracy=0.05,
       maximum_run_length=500_000,
   )

Mixed Accuracy Criteria (Three Variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = run_length_control(
       get_trajectory=multi_var_simulator,
       number_of_variables=3,
       relative_accuracy   = [0.01, 0.05, None],
       absolute_accuracy   = [None, None, 0.15],
   )

Validate Against Known Gamma Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = run_length_control(
       get_trajectory=gamma_simulator,
       relative_accuracy=0.01,
       population_cdf='gamma',
       population_args=(2.0,),   # shape
       population_loc=0.0,
       population_scale=1.0,
   )

Common Pitfalls
---------------

Mean Close to Zero with Relative Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Problem: Mean near zero makes relative accuracy unstable
   # Instead of:
   run_length_control(..., relative_accuracy=0.05)

   # Use absolute accuracy:
   run_length_control(
       ...,
       relative_accuracy=None,
       absolute_accuracy=0.1
   )

Overly Strict Accuracy Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 0.1% relative accuracy may require millions of steps
   run_length_control(..., relative_accuracy=0.001)  # Often unrealistic
   run_length_control(..., relative_accuracy=0.01)   # More practical

Performance Guide
-----------------

Trajectory Growth Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

=========== =============== ==============
Factor      # UCL calls     Overhead
=========== =============== ==============
1.2         21              1.8 s
1.5         9               0.9 s
2.0         6               0.7 s
=========== =============== ==============

.. note::
   Performance numbers are for illustration only.
   Actual overhead depends on UCL method, trajectory length, and hardware.

Memory Footprint
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Single variable: 1-D array, 8 bytes x ``total_steps``
* Multiple variables: 2-D array, 8 bytes x ``n_vars x total_steps``
* Use ``dump_trajectory=True`` to stream very long runs to disk

UCL Method Trade-off
~~~~~~~~~~~~~~~~~~~~~~~~~~

============================== =========== =========== ===========
Method                         Correlation Skewness    Speed
============================== =========== =========== ===========
``uncorrelated_sample``        Low         None        Fastest
``heidel_welch``               High        Low         Fast
``n_skart``                    High        High        Moderate
============================== =========== =========== ===========

.. note::
   - Start with ``uncorrelated_sample`` (fastest)
   - If convergence seems too slow, switch to ``heidel_welch``
   - Use ``n_skart`` for highly skewed or heavy-tailed data

Error Cheat-Sheet
-----------------

.. dropdown:: Diagnose your error in 15 seconds

   .. code-block:: text

      Simulation stops early?
      +-- equilibration_step >= maximum_equilibration_step
      |   -> increase maximum_equilibration_step or check simulator
      +-- trajectory length >= maximum_run_length
          -> loosen relative_accuracy or increase maximum_run_length

      UCL = nan?
      +-- check trajectory for NaN / Inf

Internal Layout
---------------

========================== ==================================================
Module                     Responsibility
========================== ==================================================
``_equilibration.py``      MSER truncation detection
``_convergence.py``        UCL-based accuracy checking
``_population.py``         Hypothesis tests vs. population
``_accuracy.py``           Validates user accuracy specs
``_variable_list_factory`` Normalises per-variable arguments
``_run_length.py``         Geometric trajectory extension
``_trajectory.py``         Safe data acquisition wrapper
``_setup.py``              Initial parameter validation
``core.py``                Main orchestration loop
========================== ==================================================

Science Summary
---------------

Sequential confidence intervals are recomputed on the **stationary** portion
of the trajectory only.  The effective sample size

.. math::
   N_{\text{eff}} = \frac{N}{\tau}

accounts for autocorrelation (tau = integrated autocorrelation time).  The
trajectory is extended **geometrically** to minimize the number of expensive
UCL recomputations while guaranteeing that the final interval meets the
requested coverage.

Key statistical features:

* **Joint convergence**: All variables must satisfy criteria simultaneously
* **Conservative design**: Fails early if equilibration not detected
* **Adaptive checking**: Fewer UCL computations as trajectory grows
* **Population validation**: Optional tests against known distributions
