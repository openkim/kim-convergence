"""Factory for kim-convergence-compatible get_trajectory callback."""

import numpy as np
from typing import Callable, Any

from lammps import lammps  # type: ignore[import]
import kim_convergence as cr


def build_get_trajectory(
    lmp: lammps, config: dict[str, Any], nevery: int
) -> Callable[[int, dict[str, Any]], np.ndarray]:
    r"""
    Return a closure with signature.

    get_trajectory(nstep: int, args: dict) -> ndarray
    that internally knows about lmp & config.
    """
    fix_id = "cr_fix"

    nprocs = lmp.extract_setting("world_size")
    if nprocs > 1:
        try:
            from mpi4py import MPI  # type: ignore[import]
        except ImportError as e:
            raise cr.CRError("MPI run requires mpi4py package.") from e

        comm = MPI.COMM_WORLD

    def get_trajectory(nstep: int, args: dict[str, Any]) -> np.ndarray:
        # 1.  advance simulation
        args["stop"] += nstep

        finalstep = (args["stop"] // nevery) * nevery
        if finalstep > args["stop"]:
            finalstep -= nevery
        ncountmax = (finalstep - args["initialstep"]) // nevery + 1
        args["initialstep"] = finalstep + nevery

        if nprocs > 1:
            # Send nstep to all other ranks
            for r in range(1, nprocs):
                comm.send(nstep, dest=r, tag=56789)

        # Run the LAMMPS simulation
        lmp.command(f"run {nstep}")

        var_names = config["var_names"]
        ctrl_map = config["ctrl_map"]
        n_vars = len(var_names)

        # 2.  bounds checking (if any)
        if ctrl_map:
            # trajectory array
            _ndim = config["number_of_variables"]
            trajectory = np.empty((_ndim, ncountmax), dtype=np.float64)

            # argument index in the trajectory array
            _j = 0
            for j in range(n_vars):
                var_name = var_names[j]
                if var_name in ctrl_map:
                    lb, ub = ctrl_map[var_name]
                    if lb and ub:
                        for _nstep in range(args["nstep"], args["nstep"] + ncountmax):
                            val = lmp.extract_fix(fix_id, 0, 2, _nstep, j)
                            if val <= lb or val >= ub:
                                raise cr.CRError(
                                    f'the "{var_name}"\'s value = {val} is '
                                    f"out of bound of ({lb} {ub}). This run "
                                    "is unstable."
                                )
                        continue
                    elif lb:
                        for _nstep in range(args["nstep"], args["nstep"] + ncountmax):
                            val = lmp.extract_fix(fix_id, 0, 2, _nstep, j)
                            if val <= lb:
                                raise cr.CRError(
                                    f'the "{var_name}"\'s value = {val} is '
                                    f"out of bound of ({lb} ...). This run "
                                    "is unstable."
                                )
                        continue
                    elif ub:
                        for _nstep in range(args["nstep"], args["nstep"] + ncountmax):
                            val = lmp.extract_fix(fix_id, 0, 2, _nstep, j)
                            if val >= ub:
                                raise cr.CRError(
                                    f'the "{var_name}"\'s value = {val} is '
                                    f"out of bound of (... {ub}). This run "
                                    "is unstable."
                                )
                        continue
                else:
                    for i, _nstep in enumerate(
                        range(args["nstep"], args["nstep"] + ncountmax)
                    ):
                        trajectory[_j, i] = lmp.extract_fix(fix_id, 0, 2, _nstep, j)
                    _j += 1

            args["nstep"] += ncountmax
            return trajectory.squeeze() if _ndim == 1 else trajectory

        # 3.  no bounds
        if n_vars == 1:
            trajectory = np.empty((ncountmax), dtype=np.float64)
            for i, _nstep in enumerate(range(args["nstep"], args["nstep"] + ncountmax)):
                trajectory[i] = lmp.extract_fix(fix_id, 0, 1, _nstep, 0)
            args["nstep"] += ncountmax
            return trajectory

        trajectory = np.empty((n_vars, ncountmax), dtype=np.float64)
        for j in range(n_vars):
            for i, _nstep in enumerate(range(args["nstep"], args["nstep"] + ncountmax)):
                trajectory[j, i] = lmp.extract_fix(fix_id, 0, 2, _nstep, j)
        args["nstep"] += ncountmax
        return trajectory

    return get_trajectory
