ASE Integration Module
======================

Overview
--------

The :mod:`kim_convergence.ase` module connects kim-convergence to the
`Atomic Simulation Environment (ASE) <https://ase-lib.org/>`_. It drives an
ASE ``MolecularDynamics`` object in chunks, collects one or more properties
through ASE's observer mechanism, and feeds them to
:func:`kim_convergence.run_length_control` to automatically detect
equilibration and control the run length.

.. note::

   This module is optional and requires ASE. Install it with::

      pip install ase

   or, together with kim-convergence::

      pip install kim-convergence[ase]

For a narrative, copy-paste-ready walkthrough see the
:ref:`ASE Integration example <ase-integration>`. A complete runnable script
is provided in ``examples/ase/example_equilibration.py``.

Contents
--------

.. toctree::
   :maxdepth: 2

1. :ref:`Equilibration Driver <ase-equilibration>`
2. :ref:`Property Extractors <ase-extractors>`

.. _ase-equilibration:

Equilibration Driver
--------------------

.. automodule:: kim_convergence.ase.equilibration
   :members:
   :exclude-members: __all__

.. _ase-extractors:

Property Extractors
-------------------

Built-in extractors map a property name to a callable that takes an ASE
``Atoms`` object and returns a ``float``. The available names are ``energy``
(or ``potential_energy``), ``kinetic_energy``, ``total_energy``,
``temperature``, ``volume``, ``pressure``, and ``density``. Custom extractors
can be supplied through the ``extractors`` argument of
:class:`~kim_convergence.ase.ASESampler`.

.. automodule:: kim_convergence.ase.extractors
   :members:
   :exclude-members: __all__
