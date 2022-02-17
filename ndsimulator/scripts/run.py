"""
Commandline executable
======================

usage: ndsimulator [-h] config

"""

import argparse
import logging
import numpy as np
import time

from pyfile_utils import Config
from ndsimulator.ndrun import NDRun

default_config = dict(verbose="info", run_name="ndrun", seed=0, root="./")


def main(args=None):

    start = time.time()

    # initialization
    config = parse_command_line(args)

    simulation = NDRun(**dict(config))

    if simulation.method != "2w_committor":
        simulation.begin()
        simulation.run()
        simulation.end()
    else:
        run_name = config.run_name
        simulation.method = "committor"

        x = np.copy(simulation.atoms.positions)
        dx = config.dx

        simulation.run_name = f"{run_name}_forward"
        if config.kBT == 0:
            alldx = np.random.normal(0, dx, config.ndim)
            simulation.perturb_atom(alldx=alldx)
        simulation.begin()
        v_forward = np.copy(simulation.atoms.velocities)
        logging.info(f"{simulation.atoms.positions} {simulation.atoms.velocities}")
        simulation.run()
        simulation.end()
        na = simulation.commit_basin

        simulation.run_name = f"{run_name}_backward"
        simulation.modify.set_positions(x)
        if config.kBT == 0:
            simulation.perturb_atom(alldx=-alldx)
        else:
            simulation.modify.set_velocity(v=-v_forward)
        simulation.begin()
        logging.info(f"{simulation.atoms.positions} {simulation.atoms.velocities}")
        simulation.run()
        nb = simulation.commit_basin
        logging.info(f"{na} {nb}")

    end = time.time()
    logging.info(f"total time: {end - start}")


def parse_command_line(args=None):

    parser = argparse.ArgumentParser(description="Run a ND simulation")
    parser.add_argument("config", help="configuration file")

    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)

    return config


if __name__ == "__main__":
    main()
