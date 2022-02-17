import numpy as np
from ndsimulator.data import AllData
from ndsimulator.constant import kB
from ndsimulator.engines.mc import MC


class WangLandau(MC):
    def initialize(self, run):

        MC.initialize(self, run)

        self.nbasin = 1
        self.Emin = -1.5
        self.Emax = 0.2
        grid = 101
        self.grid = grid
        self.hE = [np.zeros(grid)]
        self.Elist = [
            np.arange(grid) / float(grid - 1.0) * (self.Emax - self.Emin) + self.Emin
        ]
        self.f = [2.0]
        self.clf = None

    def copy_hE(self, hist):
        self.nbasin = hist["nbasin"]
        self.hE = []
        self.Elist = []
        self.f = []
        self.clf = hist["clf"]
        for i in range(self.nbasin):
            self.hE += [np.copy(hist["hE"][i])]
            self.Elist += [np.copy(hist["Elist"][i])]
            self.f += [hist["f"][i]]

    def dump_hE(self):
        hist = {}
        hist["nbasin"] = self.nbasin
        hist["hE"] = []
        hist["Elist"] = []
        hist["f"] = []
        hist["clf"] = self.clf
        for i in range(self.nbasin):
            hist["hE"] += [np.copy(self.hE[i])]
            hist["Elist"] += [np.copy(self.Elist[i])]
            hist["f"] += [self.f[i]]
        return hist

    def begin(self):
        MC.begin(self)

    def scaleup(self, Eid, ibasin):
        if self.hE[ibasin][Eid] == 0:
            self.hE[ibasin][Eid] += 1.0
        else:
            self.hE[ibasin][Eid] *= self.f[ibasin]

    def isflat(self):
        for i in range(self.nbasin):
            var = np.sqrt(np.variance(self.hE[i]))
            mean = np.average(self.hE[i])
            if var / mean > 0.01:
                return False
        return True

    def update(self, step, time):
        self.accept = False
        atoms = self.atoms
        pe = atoms.pe
        ndim = self.ndim
        m = self.m

        # propose a candidate

        boundary = self.bounds
        x0 = np.copy(atoms.positions)
        x = np.copy(atoms.positions)
        if self.propose_dim == "single":
            dim_new = np.random.randint(self.ndim)
            x[dim_new] += (2.0 * np.random.rand() - 1.0) * boundary[dim_new]
        elif self.propose_dim == "all":
            x += (2.0 * np.random.rand(self.ndim) - 1.0) * boundary

        if self.bound_cond == "periodic":
            x %= np.array(self.global_bounds)
        elif self.bound_cond == "no bound":
            pass

        if self.clf is None:
            ibasin = 0
            ibasin0 = 0
        else:
            ibasin = int(self.clf.predict(x.reshape([1, self.ndim]))[0])
            ibasin0 = int(self.clf.predict(x0.reshape([1, self.ndim]))[0])
        # print("ibasin, ibasin0", ibasin, ibasin0)

        pe0 = atoms.pe
        v = atoms.biase

        pe = pe0 + v
        colv = self.colvar.compute(x)

        # compare (potential) energy, kinetic = 0
        pe_new0, forces_new = self.potential.compute(x)
        vnew = 0
        if self.fixes:
            for fix in self.fixes:
                vnew += fix.energy(x0=x, col0=colv)
        pe_new = pe_new0 + vnew

        if ibasin < self.nbasin:
            hE = self.hE[0]
            Elist = self.Elist[ibasin]
            if ibasin == ibasin0:
                if pe_new > self.Emax:
                    pb = 0.0
                else:
                    Eid1 = np.argmin(np.abs(Elist - pe_new))
                    if hE[Eid1] == 0:
                        pb = 1.0
                        # print("wlmc", Eid1, hE[Eid1])
                    else:
                        Eid0 = np.argmin(np.abs(Elist - pe))
                        pb = np.min([1.0, hE[Eid0] / hE[Eid1]])
                        # print("wlmc", Eid0, Eid1, hE[Eid0], hE[Eid1])
            else:
                pb = 1.0
                Eid1 = np.argmin(np.abs(self.Elist[ibasin] - pe_new))
        else:
            pb = 1.0
            old_nbasin = int(self.nbasin)
            self.nbasin = int(ibasin + 1)
            grid = self.grid
            for i in range(old_nbasin, self.nbasin):
                self.hE += [np.zeros(grid)]
                self.Elist += [
                    np.arange(grid) / float(grid - 1.0) * (self.Emax - self.Emin)
                    + self.Emin
                ]
                self.f += [2.0]
            Eid1 = np.argmin(np.abs(self.Elist[ibasin] - pe_new))
        p = self.random.rand()
        if p <= pb:
            self.accept_rate += 1
            # print(x, atoms.positions)
            np.copyto(atoms.prev_positions, atoms.positions)
            np.copyto(atoms.positions, x)
            # TO DO, fix it. it doesn't work when colvar is svm decision function
            c = self.colvar.compute(atoms.positions)
            if c is not None:
                np.copyto(atoms.colv, self.colvar.compute(atoms.positions))
            np.copyto(atoms.forces, forces_new)
            atoms.pe = pe_new0
            atoms.ke = 0
            atoms.totale = atoms.pe
            atoms.biase = vnew
            self.accept = True
            self.scaleup(Eid1, ibasin)
        else:
            pass

        return True
