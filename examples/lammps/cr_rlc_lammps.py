"""LAMMPS entry point for kim-convergence run-length control."""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from cr_rlc_lammps_pkg.driver import Driver


def run_length_control(lmpptr, nevery: int, *argv) -> None:
    Driver(lmpptr, nevery, argv).go()
