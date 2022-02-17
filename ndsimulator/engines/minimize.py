from typing import Optional
import numpy as np
from ndsimulator.constant import *
from ndsimulator.data import AllData


class Minimize(AllData):
    def __init__(
        self,
        dt: float = 0.5,
        maxiter: Optional[int] = None,
        gtol=1.0e-5,
        beta=0.5,
        maxline=1000,
        run=None,
    ):
        super(Minimize, self).__init__(run)
        self.dt = dt * 10
        self.maxiter = maxiter
        self.gtol = gtol
        self.beta = beta
        self.maxline = maxline

    def initialize(self, run):
        AllData.__init__(self, run=run)
        if self.maxiter is None:
            self.maxiter = self.steps * 10
        self.current_dt = self.dt
        self.stop = False
        self.stop_cond = ""
        self.step = 0

    def begin(self):
        self.atoms.pe, self.atoms.forces = self.potential.compute()
        self.atoms.biase = 0
        for fix in self.fixes:
            fix_V, fix_f = fix.compute()
            self.atoms.forces += fix_f
            self.atoms.biase += fix_V

    def update(self, step=None, time=None):

        atoms = self.atoms
        v = atoms.velocities
        ndim = self.ndim
        m = self.m
        dt = self.dt

        x = atoms.positions
        pe_prev = atoms.pe + atoms.biase
        f = atoms.forces
        dx = f / m * dt * dt
        norm = np.linalg.norm
        norm_grad = norm(f)

        if atoms.prev_positions is None:
            atoms.prev_positions = np.copy(atoms.positions)
        else:
            np.copyto(atoms.prev_positions, atoms.positions)
        x += dx
        atoms.colv = self.colvar.compute(atoms.positions)
        atoms.pe, atoms.forces = self.potential.compute()
        atoms.biase = 0
        for fix in self.fixes:
            fix_V, fix_f = fix.compute()
            atoms.forces += fix_f
            atoms.biase += fix_V

        dE = atoms.pe + atoms.biase - pe_prev

        if dE >= 0:
            np.copyto(atoms.positions, atoms.prev_positions)
            self.stop_cond = "stop - dE>=0" + str(norm_grad) + " dE " + str(dE)
            self.stop = True
        if norm_grad < self.gtol:
            np.copyto(atoms.positions, atoms.prev_positions)
            self.stop_cond = "reach grad tolerance " + str(norm_grad)
            self.stop = True
        if self.step > self.maxiter:
            self.stop_cond = "stop - reach maximum iterations " + str(self.step)
            self.stop = True
        self.step += 1
