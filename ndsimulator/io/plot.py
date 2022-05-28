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

Subplot 5: pos distribution

Subplot 6: ke distribution
"""

import logging
import matplotlib.pyplot as plt

plt.set_loglevel("critical")
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
        pel=True,
        kehist=True,
        true_colvar=True,
        assumed_colvar=False,
        bias=False,
        thermo=True,
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
        self.pel_on = pel
        self.kehist_on = kehist
        self.true_colvar_on = true_colvar
        self.assumed_colvar_on = assumed_colvar
        self.thermo_on = thermo
        self.bias_on = bias
        super(Plot, self).__init__(run)

        if oneplot and movie:
            raise NameError("movie and oneplot cannot be simultaneously true")

        if oneplot:
            plt.switch_backend("agg")

    @property
    def stride(self):
        npoints = len(self.stat.pe)
        if npoints > 1000:
            s = int(np.ceil(npoints / 1000.0))
        else:
            s = 1
        return s

    def initialize(self, run):
        AllData.__init__(self, run)

        colvardim = self.true_colvar.colvardim
        if self.boundary.shape[0] != colvardim and self.boundary.shape[1] != 2:
            raise ValueError(
                f"boundary need 2 by {colvardim} elements for full plot mode"
            )
        if self.increment is None:
            self.increment = np.ones(colvardim) * (0.01)
        elif self.increment.shape[0] != colvardim:
            raise ValueError(f"increment need {colvardim} elements for full plot mode")

        if self.assumed_colvar_on:
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
        self.prev_colv = {}
        self.pos_lines = []
        self.colv_lines = {"true": [], "assumed": []}
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

    def plan_subplots(self):

        n_axes = (
            self.pel_on
            + self.kehist_on
            + self.true_colvar_on
            + self.assumed_colvar_on
            + self.bias_on
            + self.thermo_on
        )
        n_col = int(np.ceil(n_axes / 2.0))

        self.figure = plt.figure(figsize=(4 * n_col, 10))
        plt.clf()
        self.fig, self.axs = plt.subplots(2, n_col, figsize=(4 * n_col, 6))

        axs_list = np.array(self.axs).reshape([-1])

        self.ax_pos = axs_list[0]
        count = 1
        if self.kehist_on:
            self.ax_kehist = axs_list[count]
            count += 1
        if self.thermo_on:
            self.ax_thermo = axs_list[count]
            self.ax_thermo_right = self.ax_thermo.twinx()
            count += 1
        if self.true_colvar_on:
            self.ax_true_colvar = axs_list[count]
            count += 1
        if self.assumed_colvar_on:
            self.ax_assumed_colvar = axs_list[count]
            count += 1
        if self.bias_on:
            self.ax_bias = axs_list[count]
            count += 1

    def begin(self):

        self.plan_subplots()

        if self.pel_on:
            self.plot_PEL(self.ax_pos)
        self.initialize_some_colvar_pos(self.ax_pos, self.true_colvar, "true")

        if self.thermo_on and not self.oneplot:
            self.plot_thermo(1, self.ax_thermo, self.ax_thermo_right)

        # plot initial bias in colvar splace
        if self.bias_on:
            self.initialize_bias_landscape(self.ax_bias)

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

        self.update_some_colvar_pos(self.ax_pos, self.true_colvar, "true")

        if self.thermo_on:
            self.plot_thermo(self.freq, self.ax_thermo, self.ax_thermo_right)
        if self.kehist_on:
            self.plot_kehist(self.ax_kehist)

        # plotting the biased landscape
        if self.bias_on:
            self.onetimeplot_bias_landscape(self.ax_bias)

        if self.movie:
            filename = f"{self.root}/{self.run_name}/mf{self.movieframe}"
            logging.debug(f"save fig {filename}.png")
            plt.tight_layout()
            plt.savefig(f"{filename}.png", bbox_inches="tight")
            self.movieframe += 1

    def onetimeplot(self):

        # plt.clf()

        x, y = self.onetime_some_colvar(self.ax_pos, self.true_colvar, "true")

        if self.thermo_on:
            self.plot_thermo(self.freq, self.ax_thermo, self.ax_thermo_right)
        if self.bias_on:
            self.onetimeplot_bias_landscape(self.ax_bias)
        if self.true_colvar_on:
            self.plot_colvar_hist(self.ax_true_colvar, x, y, self.true_colvar, "true")
        if self.kehist_on:
            self.plot_kehist(self.ax_kehist)
        if self.assumed_colvar_on:
            x, y = self.onetime_some_colvar(
                self.ax_pos, self.colvar, "assumed", dry_run=True
            )
            self.plot_colvar_hist(self.ax_assumed_colvar, x, y, self.colvar, "assumed")

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

    def plot_PEL(self, ax):
        if self.true_colvar.colvardim == 2:
            self.plot_PEL_2d(ax)
        elif self.true_colvar.colvardim == 1:
            self.plot_PEL_1d(ax)
        else:
            raise NameError(
                f"potential energy landscape plot cannot handle {self.colvar.colvardim}-dimension colvar, please put plot_pel=false to run"
            )

    def initialize_some_colvar_pos(self, ax, colvar, name):
        pos = colvar.compute(self.atoms.positions)
        if self.oneplot is False:
            if colvar.colvardim == 2:
                x = [pos[0]]
                y = [pos[1]]
                self.prev_colv[name] = np.copy(pos)
                line = ax.plot(
                    x, y, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0
                )
            else:
                timestep = self.stat.time[-1]
                self.prev_colv[name] = [timestep, pos[0]]
                line = ax.plot(
                    timestep,
                    pos[0],
                    "o-",
                    markersize=2.5,
                    linewidth=1,
                    color="k",
                    alpha=1.0,
                )
            self.colv_lines[name].append(line)

        if self.colvar.colvardim == 2:
            ax.set_xlabel("$\\xi_1$")
            ax.set_ylabel("$\\xi_2$")
        else:
            ax.set_ylabel("$\\xi$")
            ax.set_xlabel("timestep")
        ax.set_title(f"trajectory in {name} colvar space")

    def update_some_colvar_pos(self, ax, colvar, name):
        pos = colvar.compute(self.atoms.positions)
        if colvar.colvardim == 2:
            x = [self.prev_colv[name][0], pos[0]]
            y = [self.prev_colv[name][1], pos[1]]
            for line in self.colv_lines[name]:
                old_alpha = line[0].get_alpha()
                new_alpha = old_alpha * self.decay
                if new_alpha >= 0.5:
                    line[0].set_alpha(new_alpha)
            line = ax.plot(
                x, y, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0
            )
            self.prev_colv[name] = np.copy(pos)
        elif colvar.colvardim == 1:
            x = [self.prev_colv[name][0], stat.time[-1]]
            y = [self.prev_colv[name][1], pos[0]]
            for line in self.colv_lines[name]:
                old_alpha = line[0].get_alpha()
                new_alpha = old_alpha * self.decay
                if new_alpha >= 0.5:
                    line[0].set_alpha(new_alpha)
            line = ax.plot(
                x, y, "o-", markersize=2.5, linewidth=1, color="k", alpha=1.0
            )
            self.prev_colv[name] = [stat.time[-1], pos[0]]
        self.colv_lines[name].append(line)
        # plt.colorbar(cset2)

    def onetime_some_colvar(self, ax, colvar, name, dry_run=False):

        freq = self.freq
        stride = self.stride
        x = []
        y = []
        stat = self.stat
        nconfig = int(len(stat.positions) / float(stride))
        for idx in range(nconfig):
            pos = stat.positions[idx * stride]
            colv = colvar.compute(pos)
            x += [colv[0]]
            if colvar.colvardim == 2:
                y += [colv[1]]
            else:
                y += [stat.pe[idx * stride]]
        # ax.plot(x, y, 'o-', markersize=2.5, linewidth=1,
        #                color='k', alpha=1.0)

        if not dry_run:
            ax.scatter(
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
                if colvar.colvardim == 2:
                    colv0 = colvar.compute(pos[idx * stride])
                    colv1 = colvar.compute(pos[idx * stride + stride])
                else:
                    colv0 = np.zeros([2])
                    colv1 = np.zeros([2])
                    colv0[0] = colvar.compute(pos[idx * stride])
                    colv1[0] = colvar.compute(pos[idx * stride + stride])
                    colv0[1] = stat.pe[idx * stride]
                    colv1[1] = stat.pe[idx * stride + stride]
                seg += [[[colv0[0], colv0[1]], [colv1[0], colv1[1]]]]

            coll = LineCollection(
                seg, cmap=self.OrangeAlpha, linewidths=(0.1), linestyles="solid"
            )
            coll.set_array(np.arange(len(pos) - 1))
            ax.add_collection(coll)

        return x, y

    def plot_thermo(self, freq, ax_l, ax_r):
        stat = self.stat

        stride = self.stride
        timestep = stat.time[::stride]
        pe = stat.pe[::stride]
        ke = stat.ke[::stride]
        T = stat.T[::stride]
        te = stat.totale[::stride]
        biase = stat.biase[::stride]

        if self.Tline:
            self.Tline.remove()
        else:
            ax_l.set_ylabel("temperature (K)", color="b")
            ax_l.set_xlabel("Timestep")
        (self.Tline,) = ax_l.plot(timestep, T, "b-")

        if self.peline:
            self.peline.remove()
            self.teline.remove()
            self.keline.remove()
            self.biaseline.remove()
        else:
            ax_r.set_title("energy-time")
            ax_r.set_ylabel("Energies")
            ax_r.set_xlabel("Timestep")

        (self.teline,) = ax_r.plot(timestep, te, "k-", label="total")
        (self.peline,) = ax_r.plot(timestep, pe, "r-", label="pe")
        (self.keline,) = ax_r.plot(timestep, ke, "g-", label="ke")
        (self.biaseline,) = ax_r.plot(timestep, biase, "m-", label="bias")
        ax_r.legend()

    def initialize_bias_landscape(self, ax):

        # prepare the grid
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

        self.cset3 = None
        ax.set_title("bias in assumed colvar space")
        if self.colvar.colvardim == 2:
            ax.set_xlabel("$\\xi_1$")
            ax.set_ylabel("$\\xi_2$")
        else:
            ax.set_ylabel("predicted free energy")
            ax.set_xlabel("$\\xi$")

    def onetimeplot_bias_landscape(self, ax):
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
            #     self.cset3 = ax.contourf(X, Y, bias, levels)
            #     self.colorbar3.draw_all()
            if self.cset3 is None:
                b = self.colvboundary
                self.cset3 = ax.imshow(bias, extent=b.reshape([-1]), origin="lower")
                self.colorbar3 = plt.colorbar(self.cset3, ax=ax)
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
                line = ax.plot(self.Xbias, bias, c="b", alpha=1.0)
                self.bias_lines.append(line)

                for line in self.states_lines:
                    old_alpha = line[0].get_alpha()
                    new_alpha = old_alpha * self.decay
                    if new_alpha >= 0.5:
                        line[0].set_alpha(new_alpha)
                line = ax.plot([self.atoms.colv[0]], [-batm], "bo-", alpha=1.0)
                self.states_lines.append(line)
            else:
                bias = np.zeros(self.Xbias.shape)
                for fix in self.fixes:
                    bias += fix.projection(self.Xbias, None)
                self.cset3 = ax.plot(self.Xbias, bias, "b")

    def plot_colvar_hist(self, ax, x, y, colvar, name):
        stat = self.stat
        if colvar.colvardim == 1:
            ax.hist(x, density=True, bins=100)
            ax.set_xlabel("x")
            ax.set_ylabel("counts")
        elif colvar.colvardim == 2:
            ax.hist2d(x, y, density=True)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        ax.set_title(f"{name} Colvar Dist.")

    def plot_kehist(self, ax):

        stat = self.stat
        kBT = self.kBT
        ax.hist(np.array(stat.ke) / kBT, bins=50, range=(0, 10), density=True)
        Ek = np.arange(0, 10, 0.05)
        expEk = np.exp(-Ek)
        if self.ndim == 2:
            y = expEk
        elif self.ndim == 3:
            y = expEk * 2 * np.sqrt(Ek / pi)
        elif self.ndim == 4:
            y = expEk * Ek
        elif self.ndim == 5:
            y = expEk * np.sqrt(Ek / pi) * Ek * 4 / 3.0
        if self.ndim <= 5 and self.ndim >= 2:
            ax.plot(Ek, y, "--", label="Maxwell-Boltzmann")
        ax.legend()
        ax.set_xlabel("Kinetics Energy ($k_\\mathrm{B}T$)")
        ax.set_ylabel("counts")
        ax.set_title("Kinetics Energy Dist.")
