import numpy as np
from ndsimulator.data import AllData
from ndsimulator.constant import kB


class MC(AllData):
    def __init__(
        self,
        dt=0.5,
        bounds: str = [1.0, 1.0],
        propose_dim: str = "single",  # or "all",
        global_bounds: list = [20.0, 20.0],
        bound_cond: str = "periodic",  # or "no_bound"
        run=None,
    ):
        self.dt = dt
        self.bounds = np.array(bounds)
        self.propose_dim = propose_dim
        self.global_bounds = np.array(global_bounds)
        self.bound_cond = bound_cond
        super(MC, self).__init__(run)

    def initialize(self, run):
        AllData.__init__(self, run)
        self.current_dt = self.dt

        # for mc
        self.accept_rate = 0.0
        self.accept = False

        if len(self.bounds) != self.ndim:
            raise NameError("wrong input shape of mc_bounds")

    def begin(self):
        self.atoms.pe, self.atoms.forces = self.potential.compute()
        self.atoms.biase = 0
        for fix in self.fixes:
            fix_V = fix.energy()
            # self.atoms.forces += fix_f
            self.atoms.biase += fix_V

    def update(self, step, time):
        self.accept = False
        atoms = self.atoms
        pe = atoms.pe

        # propose a candidate

        boundary = self.bounds
        x = np.copy(atoms.positions)
        if self.propose_dim == "single":
            dim_new = np.random.randint(self.ndim)
            x[dim_new] += (2.0 * np.random.rand() - 1.0) * boundary[dim_new]
        elif self.propose_dim == "all":
            x += (2.0 * np.random.rand(self.ndim) - 1.0) * boundary

        if self.bound_cond == "periodic":
            # to do, this can be tricky if the
            # initial x0 is not wrapped
            x %= np.array(self.global_bounds)
        elif self.bound_cond == "no bound":
            pass

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

        if pe_new > pe:
            pb = np.exp(-(pe_new - pe) / self.kBT)  # if pe < pe0, p0 > 1
        else:
            pb = 1.0
        p = self.random.rand()
        if p <= pb:
            # print("accept pe, penew", pe0, pe_new0, "bias", v, vnew, atoms.colv, colv)
            self.accept_rate += 1
            np.copyto(atoms.prev_positions, atoms.positions)
            np.copyto(atoms.positions, x)
            c = self.colvar.compute(atoms.positions)
            if c is not None:
                np.copyto(atoms.colv, self.colvar.compute(atoms.positions))
            # np.copyto(atoms.forces, forces_new)
            atoms.pe = pe_new0
            atoms.ke = 0
            atoms.totale = atoms.pe
            atoms.biase = vnew
            self.accept = True
        else:
            # print("reject pe, penew", pe0, pe_new0, "bias", v, vnew, atoms.colv, colv)
            pass

        return True
