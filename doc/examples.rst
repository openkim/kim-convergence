Examples
========

These snippets are **copy-paste ready**.
For complete parameter lists see :doc:`modules/run_length_control`.

.. code-block:: python

   import kim_convergence as cr
   import numpy as np


1. Single observable
--------------------

Simulate until the mean is known within 1 % relative error.

.. code-block:: python

   def get_trajectory(nstep, args):
       return np.random.normal(10, 0.5, nstep)

   report = cr.run_length_control(
       get_trajectory=get_trajectory,
       get_trajectory_args={},
       relative_accuracy=0.01,           # 1 %
       maximum_run_length=100_000,
       fp="return", fp_format="json")    # JSON string

   import json, pprint
   pprint.pp(json.loads(report))


2. Several observables, mixed accuracy
--------------------------------------

Energy (0.5 %) and pressure (2 % or ±0.05 absolute).
**All** must pass before the run stops.

.. code-block:: python

   def get_trajectory(nstep, args):
       energy   = 1000 + np.random.normal(0, 10, nstep)
       pressure =   1.0 + np.random.normal(0, 0.1, nstep)
       return np.vstack([energy, pressure])          # (2, nstep)

   out = cr.run_length_control(
       get_trajectory=get_trajectory,
       number_of_variables=2,
       relative_accuracy=[0.005, 0.02],
       absolute_accuracy=[None, 0.05],
       maximum_run_length=500_000,
       fp="return")

3. Simulator with external parameters
-------------------------------------

The **second argument must be a dict**; positional args are **not** supported.

.. code-block:: python

   def get_trajectory(nstep, args):
       T = args["temperature"]      # K
       P = args["pressure"]         # bar
       return my_sim(nstep, T=T, P=P)

   json_report = cr.run_length_control(
       get_trajectory=get_trajectory,
       get_trajectory_args={"temperature": 350, "pressure": 2.0},
       relative_accuracy=0.01,
       maximum_run_length=100_000,
       fp="return")

4. Stand-alone time-series tools
--------------------------------

Analyse *existing* data without the automatic run-length loop.

.. code-block:: python

   from kim_convergence.timeseries import (
       estimate_equilibration_length,
       statistical_inefficiency,
       uncorrelated_time_series_data_samples)

   data = np.loadtxt("output.dat")              # 1-D series

   eq, si = estimate_equilibration_length(data, nskip=10)
   clean    = data[eq:]
   si_val   = statistical_inefficiency(clean)
   uncorr   = uncorrelated_time_series_data_samples(
                   clean, si=si_val, sample_method="block_averaged")

5. OpenMM real-time callback
----------------------------

Simulation is driven in chunks; observables are collected through
a ``StateDataReporter`` writing to an in-memory buffer.

.. seealso::
   Complete working example: :download:`example1.py <../examples/openmm/example1.py>`

.. code-block:: python

   from io import StringIO
   buffer = StringIO()
   simulation.reporters.append(
       StateDataReporter(buffer, 100, totalEnergy=True, temperature=True))

   def get_traj(nstep, args):
       simulation.step(nstep)
       raw = np.genfromtxt(StringIO(buffer.getvalue()),
                           skip_header=args["skip"])
       args["skip"] += len(raw)
       return raw.T                 # (n_vars, nstep)

   state = {"skip": 1}
   report = cr.run_length_control(
       get_trajectory=get_traj,
       get_trajectory_args=state,
       number_of_variables=2,
       relative_accuracy=0.05,
       maximum_run_length=100_000,
       fp="return")

.. _lammps-integration:

6a. LAMMPS Integration
----------------------

The LAMMPS integration works through LAMMPS's built-in Python interface.

**Basic pattern in LAMMPS input script:**

.. code-block:: bash

   # Define observables as LAMMPS variables/computes
   variable natoms equal "count(all)"
   variable pea    equal "c_thermo_pe/v_natoms"

   # Call the Python function
   python run_length_control input 4 SELF 1 variable pea format piss file cr_rlc_lammps.py
   python run_length_control invoke

**Python function signature in LAMMPS context:**

.. code-block:: python

   def run_length_control(lmpptr, nevery: int, *argv) -> None:
       """Control the length of the LAMMPS simulation run.

       Args:
           lmpptr: LAMMPS pointer to a previously created LAMMPS object
           nevery: Use input values every this many timesteps
           *argv: Additional arguments specifying variables, computes, fixes,
                  and optional bounds
       """
       # Implementation creates a fix, runs simulation in chunks,
       # extracts data, and calls kim_convergence.run_length_control()

**Common usage patterns:**

Single variable:

.. code-block:: bash

   python run_length_control input 4 SELF 100 variable my_var format piss file cr_rlc_lammps.py

Multiple observables:

.. code-block:: bash

   python run_length_control input 6 SELF 10 variable energy compute temperature format pissss file cr_rlc_lammps.py

6b. LAMMPS command (input-script fragment)
------------------------------------------

Add these two lines **after** your computes/variables are defined.

.. code-block:: bash

   python run_length_control input 4 SELF 100 variable pe format piss file cr_rlc_lammps.py
   python run_length_control invoke

See :ref:`lammps-integration` above for Python implementation details.

.. _ase-integration:

7. ASE Integration
------------------

The :mod:`kim_convergence.ase` module drives an
`ASE <https://ase-lib.org/>`_ ``MolecularDynamics`` object until the
monitored property converges. It requires ASE (``pip install ase``).

.. seealso::
   Complete working example:
   :download:`example_equilibration.py <../examples/ase/example_equilibration.py>`

**Basic usage** – equilibrate a Cu system on temperature to 10 % accuracy:

.. code-block:: python

   from ase import units
   from ase.build import bulk
   from ase.calculators.emt import EMT
   from ase.md.langevin import Langevin
   from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

   from kim_convergence.ase import ASESampler, run_ase_equilibration

   atoms = bulk("Cu", cubic=True) * (4, 4, 4)
   atoms.calc = EMT()
   MaxwellBoltzmannDistribution(atoms, temperature_K=600)

   dyn = Langevin(atoms, timestep=5 * units.fs,
                  temperature_K=500, friction=0.01 / units.fs)

   sampler = ASESampler(dyn, property_names="temperature")
   result = run_ase_equilibration(
       sampler,
       initial_run_length=500,
       maximum_run_length=20000,
       relative_accuracy=0.1,          # 10 %
       confidence_coefficient=0.95)

   if result["converged"]:
       print(f"Equilibrated in {result['total_run_length']} samples")
       print(f"Mean temperature : {result['mean']}")

``run_ase_equilibration`` forwards all keyword arguments to
:func:`kim_convergence.run_length_control` and returns its result dictionary
(``get_trajectory``, ``fp``, and ``fp_format`` are reserved).

**Multiple observables** – pass a list of property names:

.. code-block:: python

   sampler = ASESampler(dyn, property_names=["energy", "temperature"])

**Sampling less frequently** – ``sample_interval`` collects a value every *N*
MD steps (so ``initial_run_length`` samples run ``N * initial_run_length``
steps):

.. code-block:: python

   sampler = ASESampler(dyn, property_names="energy", sample_interval=10)

**Built-in properties:** ``energy`` (or ``potential_energy``),
``kinetic_energy``, ``total_energy``, ``temperature``, ``volume``,
``pressure``, ``density``.

**Custom properties** – supply your own extractor (a callable taking an
``Atoms`` object and returning a float):

.. code-block:: python

   def get_max_force(atoms):
       return float(np.max(np.abs(atoms.get_forces())))

   sampler = ASESampler(
       dyn,
       property_names="max_force",
       extractors={"max_force": get_max_force})

8. Batch re-analysis of existing files
--------------------------------------

Treat a long pre-computed series as a synthetic trajectory.

.. code-block:: python

   import glob, numpy as np

   files = glob.glob("run_*.dat")
   big = np.concatenate([np.loadtxt(f) for f in files])

   json_out = cr.run_length_control(
       get_trajectory=lambda n: big[:n],
       relative_accuracy=0.01,
       maximum_run_length=len(big),
       fp="return")

Inspecting the JSON report
--------------------------

All examples above return a JSON string when ``fp="return"``.
Other formats (EDN, plain text) are available via ``fp_format``.

.. code-block:: python

   import json
   r = json.loads(json_out)
   print("converged :", r["converged"])
   print("steps     :", r["total_run_length"])
   print("equil.    :", r["equilibration_step"])
   print("eff. size :", r["effective_sample_size"])
