Getting Started
===============

This section covers installation and basic usage.

Installation
------------

Requirements
~~~~~~~~~~~~

``kim-convergence`` requires **Python 3.9 or later**. You can maintain multiple
Python versions (e.g., 2.x and 3.x) on the same system without conflict.

Installers for Linux, macOS, and Windows are available at the
`official Python website <https://www.python.org/getit/>`_.

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

`uv <https://docs.astral.sh/uv/>`_ is an all-in-one Python tool that replaces
``pip``,  ``venv``, and ``pyenv``. It is written in Rust and is 10–100x faster
than standard tools.

One of the best features of ``uv`` is that you do not need to manually manage
virtual environments; it handles them for you in the background.

**1. Create a project and environment**
If you are starting a new folder, navigate to it and run:

.. code-block:: bash

   uv init
   uv add kim-convergence

*This automatically creates a virtual environment (in a ``.venv`` folder) and
installs the package.*

**2. Running your code**
Instead of manually "activating" the environment, simply use ``uv run`` to execute
your scripts. This ensures the correct environment is always used:

.. code-block:: bash

   uv run python your_script.py

**3. Installing into an existing environment**
If you already have a virtual environment active and just want to use ``uv`` as
a fast ``pip`` replacement:

.. code-block:: bash

   uv pip install kim-convergence

.. tip::
   If you don't have ``uv`` yet, you can install it with one command:
   ``curl -LsSf https://astral.sh/uv/install.sh | sh`` on (macOS/Linux) or see
   the `official installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_.

Using pip
~~~~~~~~~

**pip** is the standard tool for installing Python packages and is included by
default with modern Python installations.

To install ``kim-convergence`` using ``pip``, run:

.. code-block:: bash

   pip install kim-convergence

.. note::
   Depending on your environment, you may need to use ``pip3`` or the Python
   module flag to ensure you are targeting the correct version:

   .. code-block:: bash

      pip3 install kim-convergence
      # OR
      python3 -m pip install kim-convergence

Using pip (GIT Support)
~~~~~~~~~~~~~~~~~~~~~~~

``pip`` also supports installing directly from the source repository via ``git``:

.. code-block:: bash

   pip install git+https://github.com/openkim/kim-convergence.git

For more information and advanced usage, see the
`pip install reference <https://pip.pypa.io/en/stable/reference/pip_install/#id18>`_.

Using pixi or conda
~~~~~~~~~~~~~~~~~~~

If you use the Conda ecosystem, we recommend **pixi** for its speed and
reproducibility. Both tools use the ``conda-forge`` channel.

**Using pixi:**

.. code-block:: bash

   pixi add kim-convergence

**Using conda:**

**conda** is the package and environment management tool used primarily with
Anaconda and Miniconda distributions.

To install ``kim-convergence`` from the ``conda-forge`` channel, first add
the channel and set the priority to ``strict``:

.. code-block:: bash

   conda config --add channels conda-forge
   conda config --set channel_priority strict

Once the channel is enabled, install the package with:

.. code-block:: bash

   conda install kim-convergence

To view all available versions of ``kim-convergence`` for your platform, use:

.. code-block:: bash

   conda search kim-convergence --channel conda-forge

Core Concepts
-------------

Simulations (e.g., molecular dynamics, Monte Carlo) often require determining
**how long to run** to achieve reliable results. Running too short leads to
bias, while too long wastes resources. ``kim-convergence`` automates this using
statistical methods to detect equilibration (steady-state) and control run
length based on desired accuracy.

The package monitors your simulation in real-time, detects when equilibration is
achieved, and continues until statistical accuracy requirements are met.

Two accuracy requirements control the simulation length:

- **Relative accuracy**: Controls the ratio of confidence interval width to mean
  value
  - Example: ``relative_accuracy=0.01`` means ±1% relative precision
  - Use when you care about percentage accuracy

- **Absolute accuracy**: Controls the absolute width of confidence intervals
  - Example: ``absolute_accuracy=0.1`` means ±0.1 absolute units
  - Use when the magnitude of your observable matters more than relative precision

See :doc:`theory` for the science behind adaptive simulation extension.

Quick Start
-----------

Here's a minimal complete example for a single variable:

.. code-block:: python

    import kim_convergence as cr
    import numpy as np

    def get_trajectory(nstep):
        # Simulate some data
        return np.random.normal(10, 0.1, nstep)

    # Run convergence control
    result = cr.run_length_control(
        get_trajectory=get_trajectory,
        relative_accuracy=0.01,
        maximum_run_length=100000,
    )

    print(f"Converged: {result}")

For simulations tracking multiple observables:

.. code-block:: python

    def get_trajectory(nstep):
        """Return 2D array: (n_variables, nstep)"""
        pressure = np.random.normal(1.0, 0.05, nstep)
        temperature = np.random.normal(300, 5, nstep)
        return np.array([pressure, temperature])

    result = cr.run_length_control(
        get_trajectory=get_trajectory,
        number_of_variables=2,
        relative_accuracy=[0.01, 0.02],  # Different accuracy for each variable
        maximum_run_length=100000,
    )

Validate Your Results
---------------------

Always check the convergence report:

.. code-block:: python

    result = cr.run_length_control(
        get_trajectory=get_trajectory,
        relative_accuracy=0.01,
        fp='return',
        fp_format='json',
    )

    import json
    details = json.loads(result)

    # Check key metrics
    print(f"Converged: {details['converged']}")
    print(f"Effective sample size: {details['effective_sample_size']}")
    print(f"Relative half-width: {details['relative_half_width']}")
    print(f"Mean: {details['mean']} ± {details['upper_confidence_limit']}")

Working with Different Simulation Packages
------------------------------------------

The package integrates with LAMMPS and OpenMM via callbacks. For custom
simulators, provide a ``get_trajectory`` function. See :doc:`examples` for code
snippets.

Best Practices
--------------

- Start with default parameters for UCL methods.
- Use FFT for large datasets to speed up autocorrelation.

For more, see :doc:`troubleshooting`.
