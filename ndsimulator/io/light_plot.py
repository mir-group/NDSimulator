"""
Light Plot
~~~~~~~~~~~~~~
This module implements the light plotting scheme

The subplot 1 and 2 are the same with the first two subplot
in the plot class.  The last two in the normal plot are not
included because they are the time consuming steps and only
works well with metadynamics, but not other sampling method yet.

* subplot 1 is for
* subplot 2 is for

Subplot 1: the trajectories of atoms at the true colvar space
    boundary (4-element list), increment (2-element list)
    and ebound (3-element list) can be used to tune the
    potential energy isosurface at the background

Subplot 2: the potential/bias/total energy and temperature over
    different timesteps.

parameters in the input file:

increment: list of numbers.  can be one or two elements
boundary: list of numbers, can be two or four elements

"""

import logging
from math import pi
import matplotlib.pyplot as plt

plt.set_loglevel("critical")
import numpy as np

from .plot import Plot


class LightPlot(Plot):
    def __init__(
        self,
        root: str,
        run_name: str,
        oneplot: bool = False,
        movie: bool = False,
        freq: int = 10,
        ebound: list = None,
        type: str = "png",
        boundary=None,
        increment=None,
        colvboundary=None,
        colvincrement=None,
        egrid=None,
        run=None,
    ):
        self.light_plot = True
        Plot.__init__(
            self,
            root=root,
            run_name=run_name,
            oneplot=oneplot,
            movie=movie,
            freq=freq,
            ebound=ebound,
            type=type,
            boundary=boundary,
            increment=increment,
            colvboundary=colvboundary,
            colvincrement=colvincrement,
            egrid=egrid,
            run=run,
        )

    def plan_subplots(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(8, 6))
        self.ax0 = self.axs[0, 0]
        self.ax1 = self.axs[0, 1]
        self.ax2 = self.ax1.twinx()
        self.ax3 = self.axs[1, 0]
        self.ax4 = self.axs[1, 1]

    def onetimeplot(self, last=False):

        x, y = self.onetime_some_colvar(self.ax0, self.true_colvar, "true")

        self.plot_thermo(self.freq, self.ax1, self.ax2)
        self.plot_colvar_hist(self.ax3, x, y, self.true_colvar, "true")
        self.plot_kehist(self.ax4)

        filename = f"{self.root}/{self.run_name}/oneplot"
        logging.debug(f"save fig {filename}.png")
        plt.tight_layout()
        plt.savefig(f"{filename}.png", bbox_inches="tight")

    def begin(self):

        self.figure = plt.figure()
        plt.clf()
        self.plan_subplots()

        self.plot_PEL(self.ax0)
        self.initialize_some_colvar_pos(self.ax0, self.true_colvar, "true")

    def update(self, last=False):
        if self.oneplot:
            return

        self.update_some_colvar_pos(self.ax0, self.true_colvar, "true")
        self.plot_thermo(self.freq, self.ax1, self.ax2)
        self.plot_kehist(self.ax4)

        if self.movie:
            filename = f"{self.root}/{self.run_name}/mf{self.movieframe}"
            logging.debug(f"save fig {filename}.png")
            plt.tight_layout()
            plt.savefig(f"{filename}.png", bbox_inches="tight")
            self.movieframe += 1
