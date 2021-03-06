from ndsimulator.data import AllData
from ndsimulator.constant import kB
import os
import numpy as np


class Modify(AllData):
    def initialize(self, run):
        AllData.__init__(self, run=run)

    def perturb_atom(self, dx=None, alldx=None):
        x0 = self.atoms.positions
        if alldx is None:
            x0 += self.random.normal(0, dx, self.ndim)
        else:
            x0 += alldx

    def perturb_velocities(self, dv):
        atoms = self.atoms
        ndim = self.ndim
        m = self.amass

        # rescale the velocity to keep the total ke
        v0 = atoms.velocities
        v0_norm = np.linalg.norm(v0)
        v0 += (self.random.rand(ndim) - 0.5) * dv
        vnew_norm = np.linalg.norm(v0)
        v0 = v0 / vnew_norm * v0_norm
        atoms.ke = np.sum(v0**2) / 2.0 * m
        atoms.T = atoms.ke * 2.0 / float(ndim * kB)
        atoms.totale = atoms.ke + atoms.pe
        self.stat.modify()

    def set_positions(self, x):
        np.copyto(self.atoms.positions, x)
        np.copyto(self.atoms.colv, self.colvar.compute(self.atoms.positions))

    def set_velocity(self, T=None, v=None):

        atoms = self.atoms
        ndim = self.ndim
        m = self.amass

        if T is not None:
            scale = np.sqrt(kB * T / m)
            u = np.array([self.random.normal(0, 1, 1) for i in range(ndim)]).reshape(
                [-1]
            )
            atoms.velocities = scale * u
            v = atoms.velocities
            mv2 = m * np.sum(v * v)
            atoms.ke = mv2 / 2.0
            atoms.T = mv2 / float(ndim * kB)
            atoms.totale = atoms.ke + atoms.pe
        elif v is not None:
            atoms.velocities = np.copy(v)
            mv2 = m * np.sum(v * v)
            atoms.ke = mv2 / 2.0
            atoms.T = mv2 / float(ndim * kB)
            atoms.totale = atoms.ke + atoms.pe
            self.stat.modify()
        else:
            raise RuntimeError("velocity need to be assigned")
