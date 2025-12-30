kim-convergence Utility Module
==============================

How Do You Automatically Estimate the Length of the Simulation Required?
------------------------------------------------------------------------

These visualizations demonstrate why manual estimation of simulation length is
difficult. The animations show trajectories from an NPT molecular dynamics
simulation tracking three key properties: temperature (T), pressure (P), and
box volume (V).

.. list-table::
   :widths: 33 33 33

   * - .. image:: files/vid1_T.gif
          :width: 200px
          :height: 200px
     - .. image:: files/vid1_P.gif
          :width: 200px
          :height: 200px
     - .. image:: files/vid1_V.gif
          :width: 200px
          :height: 200px

.. list-table::
   :widths: 33 33 33

   * - .. image:: files/vid2_T.gif
          :width: 200px
          :height: 200px
     - .. image:: files/vid2_P.gif
          :width: 200px
          :height: 200px
     - .. image:: files/vid2_V.gif
          :width: 200px
          :height: 200px

**Top row:** 10 ps simulation | **Bottom row:** 50 ps simulation

**Key observations:**

- Different properties converge at different rates
- Visual inspection cannot determine statistical reliability
- Manual estimation is therefore unreliable

It is desirable to simulate the minimum amount of time necessary to reach an
acceptable amount of uncertainty in the quantity of interest. The
`kim-convergence` package addresses this by providing tools for
**automatic equilibration detection** and **run length control** in simulations.

Key Features
------------

- **Equilibration Detection:** Identifies when your simulation reaches
  steady-state using methods like
  :ref:`Marginal Standard Error Rule (MSER) <white1997>`.

- **Run Length Control:** Extends simulations adaptively until user-specified
  accuracy is achieved, based on confidence intervals and
  :ref:`statistical inefficiency <chodera2016>`.

- **Upper Confidence Limits (UCL):** Methods like
  :ref:`Heidelberger-Welch <heidelberger1981>`,
  :ref:`N-SKART <tafazzoli2011>`,
  and :ref:`MSER-m <spratt1998>`
  for estimating uncertainty.

- **Time Series Analysis:** Tools for statistical inefficiency, autocorrelation,
  and sampling uncorrelated data.

- **Integration-Friendly:** Works with LAMMPS, OpenMM, and custom simulators via
  a callback API.

For installation and basic usage, see :doc:`getting_started`. For scientific
details, see :doc:`theory` and the Core Modules section.

