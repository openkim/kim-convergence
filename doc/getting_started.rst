Getting Started
===============

This guide helps you get up and running with ``kim-convergence`` in minutes.

Quick Start (Minimal Example)
-----------------------------

Copy-paste this complete script to see the package in action:

.. code-block:: python

    import json
    import numpy as np

    import kim_convergence as cr

    def get_trajectory(nstep):
        # Fake temperature data from a simulation (in K)
        return np.random.normal(300, 5, nstep)  # Mean 300 K, std 5 K

    if __name__ == "__main__":
        result = cr.run_length_control(
            get_trajectory=get_trajectory,
            relative_accuracy=0.01,        # Target ±1 % precision on the mean
            maximum_run_length=200_000,
            fp="return",
            fp_format="json",
        )
        details = json.loads(result)
        print(f"Converged: {details['converged']}")
        print(f"Final steps: {details['total_run_length']:,}")

For multiple observables (e.g., pressure and temperature), simply return a 2-D
array of shape ``(n_variables, nstep)`` and supply per-variable accuracies. See
the full multi-variable example (including OpenMM integration) and other cases
in :doc:`examples`.

Installation
------------

Requirements
~~~~~~~~~~~~

``kim-convergence`` requires **Python 3.9 or later**. Download installers for
all platforms from the `official Python website <https://www.python.org/getit/>`_.

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

`uv <https://docs.astral.sh/uv/>`_ is a fast, modern Python project and package
manager written in Rust (10–100x faster than pip/venv).

Inside your project folder:

.. code-block:: bash

    uv init                     # creates .venv automatically
    uv add kim-convergence

Run scripts without manual activation:

.. code-block:: bash

    uv run python your_script.py

To install into an existing environment:

.. code-block:: bash

    uv pip install kim-convergence

.. tip::
   Install ``uv`` with a single command (macOS/Linux):

   .. code-block:: bash

       curl -LsSf https://astral.sh/uv/install.sh | sh

   Full instructions:
   `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_.

Using pip
~~~~~~~~~

.. code-block:: bash

    pip install kim-convergence

.. note::
   If your system has multiple Python versions:

   .. code-block:: bash

       pip install kim-convergence
       # or
       python -m pip install kim-convergence

Install from source
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install git+https://github.com/openkim/kim-convergence.git

Using pixi or conda
~~~~~~~~~~~~~~~~~~~

**pixi** (fast and reproducible):

.. code-block:: bash

    pixi add kim-convergence

**conda**:

.. code-block:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda install kim-convergence

Core Concepts
-------------

Molecular dynamics, Monte Carlo, and other simulations need to run long enough
for statistically reliable averages — but not longer than necessary.

``kim-convergence`` automates this by:

- Detecting **equilibration** (when the system reaches steady-state)
- Extending the run until user-defined **statistical accuracy** is achieved

Accuracy is controlled by one or both of:

**Relative accuracy**
   Half-width of the confidence interval divided by the mean.
   Example: ``relative_accuracy=0.01`` → ±1 % precision.

**Absolute accuracy**
   Absolute half-width of the confidence interval.
   Example: ``absolute_accuracy=0.1`` → ±0.1 units.

The algorithm stops only when **all** specified observables meet their criteria.
Full details are in :doc:`theory`.

Validate Your Results
---------------------

Always examine the detailed convergence report:

.. code-block:: python

    result = cr.run_length_control(
        get_trajectory=get_trajectory,
        relative_accuracy=0.01,
        fp="return",
        fp_format="json",
    )

    import json, pprint
    pprint.pp(json.loads(result))

Key fields include ``converged``, ``total_run_length``, ``equilibration_step``,
``effective_sample_size``, ``mean``, and ``upper_confidence_limit``.

Integration with Simulation Packages
------------------------------------

- **LAMMPS** and **OpenMM** are supported via dedicated callbacks (see
  :doc:`examples` for full code).
- For **custom simulators**, simply provide a ``get_trajectory`` callable that
  returns a NumPy array.
- Enable ``dump_trajectory=True`` to save raw data for debugging if convergence
  fails.

Complete integration examples (including real-time callbacks and post-processing
of existing trajectories) are in :doc:`examples`.

Best Practices & Next Steps
---------------------------

- Start with default parameters — they work well for most cases.
- Use FFT-based autocorrelation (enabled by default) for large datasets.
- Always inspect the JSON report to confirm physical reasonableness.
- For tuning advice, common issues, and advanced usage, see
  :doc:`best_practices` and :doc:`troubleshooting`.

Full API reference: :doc:`api`.
