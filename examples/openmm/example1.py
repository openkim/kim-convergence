"""Run length control example for OpenMM."""

import convergence as cr
from io import StringIO
import numpy as np
from sys import stdout

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

# use values every this many timesteps.
nevery = 100
buffer = StringIO()

# OpenMM setup
pdb = PDBFile('input.pdb')
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology,
                                 nonbondedMethod=PME,
                                 nonbondedCutoff=1*nanometer,
                                 constraints=HBonds)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond,
                                      0.004*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(
    StateDataReporter(buffer,
                      nevery,
                      totalEnergy=True,
                      temperature=True,
                      separator='\t'))


def get_trajectory(step: int, args: dict) -> np.ndarray:
    """Get trajectory vector or array of values.

    Arguments:
        step (int): number of steps to run the simulation.

    Returns:
        ndarray: trajectory
            for a single specified value, the values are stored as
            a vector. For multiple specified values, they are stored as
            rows in an array.

    """
    args['stop'] += step
    simulation.step(step)
    trajectory = np.genfromtxt(StringIO(buffer.getvalue()),
                               skip_header=args['skip_header'])
    args['skip_header'] += len(trajectory)
    if trajectory.ndim == 1:
        return trajectory
    return trajectory.transpose()


get_trajectory_args = {
    'stop': 0,
    'skip_header': 1,
}


try:
    msg = cr.run_length_control(
        get_trajectory=get_trajectory,
        get_trajectory_args=get_trajectory_args,
        number_of_variables=2,
        initial_run_length=2000,
        run_length_factor=1.0,
        maximum_run_length=100000,
        maximum_equilibration_step=20000,
        minimum_number_of_independent_samples=20,
        relative_accuracy=0.1,
        absolute_accuracy=None,
        confidence_coefficient=0.95,
        confidence_interval_approximation_method='uncorrelated_sample',
        dump_trajectory=True,
        dump_trajectory_fp='convergence_trajectory.edn',
        fp='return',
        fp_format='txt')
except Exception as e:
    msg = '{}'.format(e)
    raise cr.CVGError(msg)

print(msg)
