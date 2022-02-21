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

    def begin(self):

        self.figure = plt.figure()
        plt.clf()

        self.prepare_subplot1()
        self.prepare_subplot2()

    def update(self, last=False):
        if self.oneplot:
            return

        ax0 = self.ax0
        timestep = self.stat.time
        cond1 = timestep[-1] > 0 and self.prev_timestep >= 0
        cond2 = timestep[-1] < 0 and self.prev_timestep <= 0
        if cond1 or cond2:
            if self.true_colvar.colvardim == 2:
                true_colv = self.true_colvar.compute(self.atoms.positions)
            else:
                true_colv = [
                    self.true_colvar.compute(self.atoms.positions),
                    self.atoms.pe,
                ]
            x = np.vstack([self.prev_pos[0], true_colv[0]])
            y = np.vstack([self.prev_pos[1], true_colv[1]])
        else:
            true_colv = self.true_colvar.compute(self.atoms.positions)
            x = [true_colv[0]]
            if self.true_colvar.colvardim == 2:
                y = [true_colv[1]]
            else:
                y = self.atoms.pe

        if timestep[-1] >= 0:
            scatter_color = "k"
        else:
            scatter_color = "r"

        line = ax0.plot(
            x, y, "o-", markersize=2.5, linewidth=1, color=scatter_color, alpha=1.0
        )
        self.pos_lines.append(line)
        self.prev_pos = [x, y]
        self.prev_timestep = self.stat.time[-1]

        if self.ax5:
            ax5 = self.ax5
            pos = self.atoms.positions
            line = ax5.scatter(pos[0], pos[1])

        self.onetimeplot_subplot2(self.freq)

        if self.movie:
            filename = f"{self.root}/{self.run_name}/mf{self.movieframe}"
            logging.debug(f"save fig {filename}.png")
            plt.tight_layout()
            plt.savefig(f"{filename}.png", bbox_inches="tight")
            self.movieframe += 1

    def onetimeplot(self, last=False):

        x, y = self.onetimeplot_subplot1()
        self.onetimeplot_subplot2(self.freq)

        self.ax3 = self.axs[1, 0]
        ax3 = self.ax3
        self.ax4 = self.axs[1, 1]
        ax4 = self.ax4

        stat = self.stat
        if self.true_colvar.colvardim == 1:
            ax3.hist(x, density=True)
            ax3.set_xlabel("x")
            ax3.set_ylabel("counts")
        elif self.true_colvar.colvardim == 2:
            ax3.hist2d(x, y, density=True)
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
        kBT = self.kBT
        ax4.hist(np.array(stat.ke) / kBT, bins=50, range=(0, 10), density=True)
        Ek = np.arange(0, 10, 0.05)
        expEk = np.exp(-Ek)
        if self.ndim == 2:
            y = expEk
            ax4.plot(Ek, y, "--")
        elif self.ndim == 3:
            y = expEk * 2 * np.sqrt(Ek / pi)
            ax4.plot(Ek, y, "--")
        elif self.ndim == 4:
            y = expEk * Ek
            ax4.plot(Ek, y, "--")
        elif self.ndim == 5:
            y = expEk * np.sqrt(Ek / pi) * Ek * 4 / 3.0
            ax4.plot(Ek, y, "--")
        ax4.set_xlabel("Kinetics Energy ($k_\\mathrm{B}T$)")
        ax4.set_ylabel("counts")

        filename = f"{self.root}/{self.run_name}/oneplot"
        logging.debug(f"save fig {filename}.png")
        plt.tight_layout()
        plt.savefig(f"{filename}.png", bbox_inches="tight")
