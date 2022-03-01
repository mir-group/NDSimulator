"""
Commandline executable
======================

usage: ndPEL [-h] kwargs_file

A command line executable to plot potential energy landscape

An example input file:

.. highlight:: yaml
.. code-block:: yaml

   potential: DoubleWell
   boundary:
   - - 10.0
     - 30.0
   - - 0.0
     - 40.0
   ebound:
   - -0.6
   - -0.2
   increment:
   - 0.1
   - 0.1

"""

import argparse
from pyfile_utils import Config, instantiate

import matplotlib.pyplot as plt

from ndsimulator.potentials import potential_from_config
from ndsimulator.io.plot import Plot


def main(args=None):

    parser = argparse.ArgumentParser(description="Plot a potential")
    parser.add_argument("kwargs", default="None", type=str)
    args = parser.parse_args(args=args)

    config = Config.from_file(args.kwargs)
    config.oneplot = False
    config.root = "./"
    config.run_name = "-"

    potential = potential_from_config(config)
    plot, _ = instantiate(Plot, prefix="plot", optional_args=config)
    plot.potential = potential
    figure, ax = plt.subplots(figsize=(3.4, 2.5))
    if config.ndim == 2:
        plot.plot_PEL_2d(ax)
    elif config.ndim == 1:
        plot.plot_PEL_1d(ax)
    figure.savefig(f"{type(potential).__name__}.png")

    return args


if __name__ == "__main__":
    main()
