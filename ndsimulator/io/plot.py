"""
Plot
~~~~

This module implements the plotting scheme to visualize simulation.

The whole plotting scheme only works if the true_colvar
output is two dimensional.

Subplot 1: the trajectories of atoms at the true colvar space
    boundary (4-element list), increment (2-element list)
    and plot_ebound (3-element list) can be used to tune the
    potential energy isosurface at the background

Subplot 2: the potential/bias/total energy and temperature over
    different timesteps.

Subplot 3: the trajectories of atoms at the guess colvar space

Subplot 4: the negative of bias energy at the colvar space
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

from ndsimulator.data import AllData
from ndsimulator.constant import kB


class Plot(AllData):
    def __init__(
        self,
        root: str,
        run_name: str,
        boundary,
        oneplot=False,
        movie=False,
        freq=10,
        ebound=None,
        type="png",
        increment=None,
        colvboundary=None,
        colvincrement=None,
        egrid=None,
        run=None,
    ):
        self.root = root
        self.run_name = run_name
        self.movie = movie
        self.freq = freq
        self.ebound = ebound
        self.type = type
        self.boundary = np.array(boundary)
        self.increment = None if increment is None else np.array(increment)
        self.colvboundary = (
            self.boundary if colvboundary is None else np.array(colvboundary)
        )
        self.colvincrement = (
            self.increment if colvincrement is None else np.array(colvincrement)
        )
        self.egrid = egrid
        self.oneplot = oneplot
        self.light_plot = False
        super(Plot, self).__init__(run)

        if oneplot and movie:
            raise NameError("movie and oneplot cannot be simultaneously true")

        if oneplot:
            plt.switch_backend("agg")

    def initialize(self, run):
        AllData.__init__(self, run)

        colvardim = self.true_colvar.colvardim
        if self.boundary.shape[0] != colvardim and self.boundary.shape[1] != 2:
            raise ValueError(
                f"boundary need 2 by {colvardim} elements for full plot mode"
            )
        if self.increment is None:
            self.increment = np.ones(colvardim) * 0.01
        elif self.increment.shape[0] != colvardim:
            raise ValueError(f"increment need {colvardim} elements for full plot mode")

        if not self.light_plot:
            colvardim = self.colvar.colvardim
            if (
                self.colvboundary.shape[0] != self.colvar.colvardim
                and self.colvboundary.shape[1] != 2
            ):
                raise ValueError(
                    f"colvboundary need 2 by {colvardim} elements for full plot mode"
                )
            if self.colvincrement is None:
                self.colvincrement = np.ones(colvardim) * 0.01
            elif self.colvincrement.shape[0] != colvardim:
                raise ValueError(
                    f"colvincrement need {colvardim} elements for full plot mode"
                )

        if self.oneplot and self.dump:
            self.stat.track_everything()

        self.peline = None
        self.keline = None
        self.teline = None
        self.biaseline = None
        self.Tline = None

        self.prev_pos = None
        self.prev_colv = None
        self.pos_lines = []
        self.colv_lines = []
        self.bias_lines = []
        self.states_lines = []
        self.decay = 0.9

        # prepare the contour plot levels
        if self.movie:
            self.movieframe = 0

        cdict = {
            "red": ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
            "green": ((0.0, 0.0, 30.0 / 255.0), (1.0, 30.0 / 255.0, 1.0)),
            "blue": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            "alpha": ((0.0, 0.0, 0.0), (1.0, 1.0, 0.95)),
        }
        self.OrangeAlpha = LinearSegmentedColormap("BlueRed1", cdict)

    def begin(self):
        self.figure = plt.figure(figsize=(12, 10))
        plt.clf()
        # plt.ion()

        self.prepare_subplot1()
        self.prepare_subplot2()

        # prepare the grid for plot 4
        cgrid = self.colvincrement
        cb = self.colvboundary
        if self.colvar.colvardim == 2:
            cX = np.arange(cb[0, 0], cb[0, 1], cgrid[0])
            cY = np.arange(cb[1, 0], cb[1, 1], cgrid[1])
            cX, cY = np.meshgrid(cX, cY)
            self.cX = cX
            self.cY = cY
            bias = np.zeros(cX.shape)
            for fix in self.fixes:
                bias += fix.projection(cX, cY)
        elif self.colvar.colvardim == 1:
            self.Xbias = np.arange(cb[0, 0], cb[0, 1], cgrid[0])
            bias = np.zeros(self.Xbias.shape[0])
            for fix in self.fixes:
                bias += fix.projection(self.Xbias, None)
        else:
            raise NameError(
                "plot can only handle 1d or 2d colvar, please put plot=false to run"
            )

        # plot initial position in colvar splace in suplot 3
        self.ax3 = self.axs[1, 0]
        ax3 = self.ax3
        if self.oneplot is False:
            if self.colvar.colvardim == 2:
                x = [self.atoms.colv[0]]
                y = [self.atoms.colv[1]]
                self.prev_colv = np.copy(self.atoms.colv)
                line = ax3.plot(
                    x, y, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0
                )
            else:
                x = self.atoms.colv[0]
                timestep = self.stat.time[-1]
                self.prev_colv = [timestep, self.atoms.colv[0]]
                line = ax3.plot(
                    timestep, x, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0
                )
            self.colv_lines.append(line)
        if self.colvar.colvardim == 2:
            ax3.set_xlabel("colvar1")
            ax3.set_ylabel("colvar2")
        else:
            ax3.set_ylabel("colvar1")
            ax3.set_xlabel("timestep")
        ax3.set_title("trajectory in colvar space")

        # plot initial bias in colvar splace in suplot 4
        self.ax4 = self.axs[1, 1]
        ax4 = self.ax4
        self.cset3 = None
        ax4.set_title("bias")
        if self.colvar.colvardim == 2:
            ax4.set_xlabel("colvar1")
            ax4.set_ylabel("colvar2")
        else:
            ax4.set_ylabel("predicted free energy")
            ax4.set_xlabel("colvar1")

        if self.oneplot is False:
            fmt = self.type
            logging.debug(f"save fig {self.root}/{self.run_name}/initial.{fmt}")
            plt.tight_layout()
            plt.savefig(
                f"{self.root}/{self.run_name}/initial.{fmt}", bbox_inches="tight"
            )

        # plt.pause(0.0000001)

    def update(self, last=False):

        if self.oneplot:
            return

        ax0 = self.ax0

        true_colv = self.true_colvar.compute(self.atoms.positions)
        x = np.hstack([self.prev_pos[0], true_colv[0]])
        y = np.hstack([self.prev_pos[1], true_colv[1]])

        for line in self.pos_lines:
            old_alpha = line[0].get_alpha()
            new_alpha = old_alpha - self.decay
            if new_alpha >= 0.5:
                line[0].set_alpha(new_alpha)
        line = ax0.plot(x, y, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0)
        self.pos_lines.append(line)
        self.prev_pos = np.copy(true_colv)

        self.onetimeplot_subplot2(self.freq)
        self.onetimeplot_subplot4()

        # plotting the biased landscape
        ax3 = self.ax3
        if self.colvar.colvardim == 2:
            x = [self.prev_colv[0], self.atoms.colv[0]]
            y = [self.prev_colv[1], self.atoms.colv[1]]
            for line in self.colv_lines:
                old_alpha = line[0].get_alpha()
                new_alpha = old_alpha * self.decay
                if new_alpha >= 0.5:
                    line[0].set_alpha(new_alpha)
            line = ax3.plot(
                x, y, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0
            )
            self.prev_colv = np.copy(self.atoms.colv)
        elif self.colvar.colvardim == 1:
            x = [self.prev_colv[0], stat.time[-1]]
            y = [self.prev_colv[1], self.atoms.colv[0]]
            for line in self.colv_lines:
                old_alpha = line[0].get_alpha()
                new_alpha = old_alpha * self.decay
                if new_alpha >= 0.5:
                    line[0].set_alpha(new_alpha)
            line = ax3.plot(
                x, y, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0
            )
            self.prev_colv = [stat.time[-1], stat.atoms.colv[0]]
        self.colv_lines.append(line)
        # plt.colorbar(cset2)

        if self.movie:
            filename = f"{self.root}/{self.run_name}/mf{self.movieframe}"
            logging.debug(f"save fig {filename}.png")
            plt.tight_layout()
            plt.savefig(f"{filename}.png", bbox_inches="tight")
            self.movieframe += 1

    def onetimeplot(self):

        # plt.clf()

        self.onetimeplot_subplot1()
        self.onetimeplot_subplot2(self.freq)
        self.onetimeplot_subplot4()

        # plotting the biased landscape
        freq = self.freq
        npoints = len(self.stat.pe)
        if npoints > 1000:
            stride = int(npoints / 1000.0)
        else:
            stride = 1
        ax3 = self.ax3
        stat = self.stat
        x = []
        y = []
        timestep = self.stat.time[::stride]
        if self.true_colvar.colvardim == 2:
            for pos in stat.positions[::stride]:
                true_colv = self.true_colvar.compute(pos)
                x += [true_colv[0]]
                y += [true_colv[1]]
            line = ax3.plot(
                x, y, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0
            )
        elif self.true_colvar.colvardim == 1:
            for pos in stat.positions[::stride]:
                colv = self.colvar.compute(pos)
                x += [colv[0]]
            line = ax3.plot(
                timestep, x, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0
            )

        fmt = self.type
        filename = f"{self.root}/{self.run_name}/oneplot"
        logging.debug(f"save fig {filename}.{fmt}")
        plt.tight_layout()
        plt.savefig(f"{filename}.{fmt}", bbox_inches="tight")
        logging.debug("end saving")

    def end(self):
        fmt = self.type
        if self.oneplot:
            self.onetimeplot()
        if self.movie:
            os.system(
                f"ffmpeg -r 1 -i {self.root}/{self.run_name}/mf%d.png -vcodec mpeg4 -y {self.root}/{self.run_name}/movie.mp4"
            )
        plt.close()

    def plot_PEL_2d(self, ax):
        # plot potential energy contour
        grid = self.increment
        b = self.boundary
        X = np.arange(b[0, 0], b[0, 1], grid[0])
        Y = np.arange(b[1, 0], b[1, 1], grid[1])
        X, Y = np.meshgrid(X, Y)
        self.X = X
        self.Y = Y
        self.original_pe = self.potential.projection(X, Y)

        if self.ebound is None:
            self.ebound = np.zeros(2)
            ave = np.average(self.original_pe)
            std = np.std(self.original_pe)
            self.ebound[0] = ave - 3 * std
            self.ebound[1] = ave + 3 * std
            if self.egrid is None:
                self.egrid = std / 2.0
        ebound = self.ebound

        if self.egrid is None:
            self.egrid = (ebound[1] - ebound[0]) / 10

        self.levels = np.arange(ebound[0], ebound[1], self.egrid)

        cset1 = ax.contourf(X, Y, self.original_pe, self.levels)
        plt.colorbar(cset1, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def plot_PEL_1d(self, ax):
        # plot potential energy contour
        grid = self.increment
        b = self.boundary
        X = np.arange(b[0, 0], b[0, 1], grid[0])
        self.X = X
        self.original_pe = self.potential.projection(X)
        ax.plot(X, self.original_pe)
        ax.set_ylabel("pe")

    def prepare_subplot1(self):

        if self.ndim == 2:
            self.fig, self.axs = plt.subplots(2, 3, figsize=(12, 6))
            self.ax5 = self.axs[0, 2]
        else:
            self.fig, self.axs = plt.subplots(2, 2, figsize=(8, 6))
            self.ax5 = None

        self.ax0 = self.axs[0, 0]
        ax0 = self.ax0
        pos = self.atoms.positions

        # plot potential energy contour
        grid = self.increment
        b = self.boundary
        if self.true_colvar.colvardim == 2:
            self.plot_PEL_2d(ax0)
            true_colv = self.true_colvar.compute(pos)
            x = [true_colv[0]]
            y = [true_colv[1]]
        elif self.true_colvar.colvardim == 1:
            self.plot_PEL_1d(ax0)
            x = self.true_colvar.compute(pos)
            y = self.atoms.pe
        else:
            raise NameError(
                f"light plot cannot handle {self.colvar.colvardim}-dimension colvar, please put plot=false to run"
            )

        line = ax0.plot(x, y, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0)
        self.pos_lines.append(line)

        self.prev_pos = [x, y]
        self.prev_timestep = self.stat.time[-1]
        ax0.set_xlabel("x")
        ax0.set_title("trajectory in real colvar space")

        if self.ax5:
            self.ax5.scatter(pos[0], pos[1], color="k")

    def prepare_subplot2(self):

        self.ax1 = self.axs[0, 1]
        self.ax2 = self.ax1.twinx()

        if not self.oneplot:
            self.onetimeplot_subplot2(1)

    def onetimeplot_subplot1(self):

        freq = self.freq
        npoints = len(self.stat.pe)
        if npoints > 1000:
            stride = int(npoints / 1000.0)
        else:
            stride = 1
        ax0 = self.ax0
        x = []
        y = []
        stat = self.stat
        nconfig = int(len(stat.positions) / float(stride))
        for idx in range(nconfig):
            pos = stat.positions[idx * stride]
            true_colv = self.true_colvar.compute(pos)
            x += [true_colv[0]]
            if self.true_colvar.colvardim == 2:
                y += [true_colv[1]]
            else:
                y += [stat.pe[idx * stride]]
        # ax0.plot(x, y, 'o-', markersize=2.5, linewidth=1,
        #                color='k', alpha=1.0)

        ax0.scatter(
            x,
            y,
            c=np.arange(len(x)),
            cmap=self.OrangeAlpha,
            linewidths=0.1,
            edgecolors="k",
        )
        seg = []
        pos = stat.positions
        for idx in range(nconfig - 1):
            if self.true_colvar.colvardim == 2:
                colv0 = self.true_colvar.compute(pos[idx * stride])
                colv1 = self.true_colvar.compute(pos[idx * stride + stride])
            else:
                colv0 = np.zeros([2])
                colv1 = np.zeros([2])
                colv0[0] = self.true_colvar.compute(pos[idx * stride])
                colv1[0] = self.true_colvar.compute(pos[idx * stride + stride])
                colv0[1] = stat.pe[idx * stride]
                colv1[1] = stat.pe[idx * stride + stride]
            seg += [[[colv0[0], colv0[1]], [colv1[0], colv1[1]]]]

        coll = LineCollection(
            seg, cmap=self.OrangeAlpha, linewidths=(0.1), linestyles="solid"
        )
        coll.set_array(np.arange(len(pos) - 1))
        ax0.add_collection(coll)

        pos = np.array(stat.positions)
        if self.ax5 and len(pos):
            pe = stat.pe
            self.ax5.scatter(pos[:, 0], pos[:, 1], c=pe)
        return x, y

    def onetimeplot_subplot2(self, freq):
        stat = self.stat
        npoints = len(self.stat.pe)
        if npoints > 1000:
            stride = int(npoints / 1000.0)
        else:
            stride = 1

        timestep = stat.time[::stride]
        pe = stat.pe[::stride]
        ke = stat.ke[::stride]
        T = stat.T[::stride]
        te = stat.totale[::stride]
        biase = stat.biase[::stride]

        ax2 = self.ax2
        if self.Tline:
            self.Tline.remove()
        else:
            ax2.set_ylabel("temperature (K)", color="b")
        (self.Tline,) = ax2.plot(timestep, T, "b-")

        ax1 = self.ax1
        if self.peline:
            self.peline.remove()
            self.teline.remove()
            self.keline.remove()
            self.biaseline.remove()
        else:
            ax1.set_title("energy-time")
            ax1.set_xlabel("time")
            ax1.set_ylabel("pe or total e")

        (self.peline,) = ax1.plot(timestep, pe, "r-", label="pe")
        (self.keline,) = ax1.plot(timestep, ke, "g-", label="ke")
        (self.teline,) = ax1.plot(timestep, te, "k-", label="total")
        (self.biaseline,) = ax1.plot(timestep, biase, "m-", label="bias")
        ax1.legend()

    def onetimeplot_subplot4(self):
        ax4 = self.ax4
        if self.colvar.colvardim == 2:
            X = self.cX
            Y = self.cY
            bias = np.zeros(X.shape)
            for fix in self.fixes:
                bias += fix.projection(X, Y)
            emin = np.min(bias)
            emax = np.max(bias)
            levels = np.arange(emin, emax, self.egrid)
            # if (levels):
            #     self.cset3 = ax4.contourf(X, Y, bias, levels)
            #     self.colorbar3.draw_all()
            if self.cset3 is None:
                b = self.colvboundary
                self.cset3 = ax4.imshow(bias, extent=b.reshape([-1]), origin="lower")
                self.colorbar3 = plt.colorbar(self.cset3, ax=ax4)
            else:
                self.cset3.set_data(bias)
                self.colorbar3.draw_all()
        elif self.colvar.colvardim == 1:
            if not self.oneplot:
                bias = np.zeros(self.Xbias.shape)
                batm = 0
                for fix in self.fixes:
                    bias += fix.projection(self.Xbias, None)
                    batm += fix.energy(col0=self.atoms.colv)

                for line in self.bias_lines:
                    old_alpha = line[0].get_alpha()
                    new_alpha = old_alpha * self.decay
                    if new_alpha >= 0.5:
                        line[0].set_alpha(new_alpha)
                line = ax4.plot(self.Xbias, bias, c="b", alpha=1.0)
                self.bias_lines.append(line)

                for line in self.states_lines:
                    old_alpha = line[0].get_alpha()
                    new_alpha = old_alpha * self.decay
                    if new_alpha >= 0.5:
                        line[0].set_alpha(new_alpha)
                line = ax4.plot([self.atoms.colv[0]], [-batm], "bo-", alpha=1.0)
                self.states_lines.append(line)
            else:
                bias = np.zeros(self.Xbias.shape)
                for fix in self.fixes:
                    bias += fix.projection(self.Xbias, None)
                self.cset3 = ax4.plot(self.Xbias, bias, "b")
