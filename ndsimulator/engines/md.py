"""
Molecular Dynamics
~~~~~~~~~~~~~~~~~~
This module implements the molecular dynamics integrator with
micro-canonical ensemble (NVE), first and second order langevin,
and velocity rescale methods.

The velocity Verlet method is used. See Eq. 3.10.31, 3.10.32 in
P111-P112 of Statistical Mechanics: Theory and Molecular
Simulation by Mark E. Tuckerman.

.. math::

   x(\Delta t) = x(0) + \Delta t v(0) + {{\Delta t^2}\over{2m}} F(x(0)) 

.. math::

   v(\Delta t) = v(0) + {{\Delta t}\over{2m}} [F(x(0))+F(x(\Delta t))]

The first and second order langevin integrator see 

  1. Chapter 15 of Statistical Mechanics: Theory and Molecular Simulation by Mark
  E. Tuckerman 

  2. Eq. 23 in E. V.-Eijnden, and G. Ciccotti, Chem.  Phys. Lett. 429, 310 (2006)

TO DO: implement dt/rescale method
"""


import numpy as np
from ndsimulator.data import AllData
from ndsimulator.constant import kB
from math import floor
from typing import Optional


class MD(AllData):
    """
    Args:
        dt (float, optional): timestep. Defaults to 0.5.
        gamma (float, optional): gamma for langevin dynamics. Defaults to 0.005.
        integrate (str, optional): integration method, langevin, 2nd-langevin, nve, and rescale. Defaults to "langevin".
        rescale_freq (Optional[int], optional): _description_. Defaults to None.
        run (AllData, optional): AllD. Defaults to None.
    """

    def __init__(
        self,
        dt: float = 0.5,
        gamma: float = 0.005,
        integrate: str = "langevin",
        rescale_freq: Optional[int] = None,
        run: AllData = None,
    ):
        self.dt = dt

        self.integrate = integrate
        assert integrate in ["langevin", "2nd-langevin", "rescale", "nve"]

        self.gamma = gamma
        self.rescale_freq = rescale_freq

        super(MD, self).__init__(run)

    def initialize(self, run):
        AllData.__init__(self, run)
        self.current_dt = self.dt

        dt = self.dt
        # for langevin
        if self.integrate == "langevin":
            self.c1, self.c2 = self.getnewc(dt)
        elif self.integrate == "2nd-langevin":
            self.c1, self.c2, self.c3, self.c4, self.c5 = self.getnewc(dt)
        elif self.integrate == "rescale":
            self.rescale_count = 0
            self.v_target2 = self.kBT * self.ndim / self.m
            self.v_target = np.sqrt(self.v_target2)

    def getnewc(self, dt):
        kBT = self.kBT
        if self.integrate == "langevin":
            c1 = np.exp(-self.gamma * dt / 2.0)
            c2 = np.sqrt((1 - c1 ** 2) * kBT / self.m)
            return c1, c2
        elif self.integrate == "2nd-langevin":
            frdt = (1.0 - np.exp(-self.gamma * dt / 2.0)) * 8.0
            m = self.m
            sigma = np.sqrt(2 * kBT * frdt / m)
            ndim = self.ndim
            c1 = dt / 2.0 - dt * frdt / 8.0
            c2 = frdt / 2 - frdt ** 2 / 8.0
            c3 = np.sqrt(dt) * sigma / 2.0 * (1.0 - frdt / 4.0)
            c5 = np.sqrt(dt) * sigma / (2 * np.sqrt(ndim))
            c4 = frdt / 2.0 * c5
            return c1, c2, c3, c4, c5
        else:
            return None

    def begin(self):
        atoms = self.atoms
        atoms.pe, atoms.forces = self.potential.compute()
        atoms.biase = 0
        atoms.fixf = np.zeros(self.ndim)
        for fix in self.fixes:
            fix_V, fix_f = fix.compute()
            atoms.fixf += fix_f
            atoms.biase += fix_V
        atoms.totale = atoms.ke + atoms.pe + atoms.biase

    def update(self, step, time):

        atoms = self.atoms
        ndim = self.ndim
        m = self.m
        if self.integrate == "langevin":
            c1 = self.c1
            c2 = self.c2
            xi = self.random.normal(0, 1, ndim)
        elif self.integrate == "2nd-langevin":
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4
            c5 = self.c5
            xi = self.random.normal(0, 1, ndim)
            eta = self.random.normal(0, 1, ndim)

        dE = 0.5
        dt = self.dt
        v0 = np.copy(atoms.velocities)
        f0 = atoms.forces + atoms.fixf
        x0 = atoms.positions
        totale0 = atoms.totale
        v = np.copy(v0)
        f = np.copy(f0)

        # # need to be fixed
        # if not self.fixdt:
        #     dtnew = np.zeros(self.ndim)
        #     for idx in range(self.ndim):
        #         a = f[idx]/m
        #         temperature= v[idx]*v[idx]-2.0*a*0.1
        #         if (temperature>=0):
        #             if (a>0):
        #                 dtnew[idx] = (np.sqrt(temperature)+v[idx])/(a+1e-25)
        #             else:
        #                 dtnew[idx] = (-np.sqrt(temperature)+v[idx])/(a+1e-25)
        #         else:
        #             dtnew[idx] = 0
        #     dt = np.min(dtnew)
        #     if (dt <= 0):
        #         raise NameError('the potential energy landscape is chaning too drastically. It is hard to find a good dt', dtnew)
        #     elif (dt > self.dt):
        #         dt = self.dt

        # if (dt != self.dt):
        #     if (self.integrate == "langevin"):
        #         c1, c2 = self.getnewc(dt)
        #     elif (self.integrate == "2nd-langevin"):
        #         c1, c2, c3, c4, c5 = self.getnewc(dt)

        if self.integrate == "langevin":
            v += dt / 2.0 * f / m
            v = c1 * v + c2 * xi
            dx = v * dt
        elif self.integrate == "nve":
            v += dt / 2.0 * f / m
            dx = v * dt
        elif self.integrate == "2nd-langevin":
            v += c1 * f / m - c2 * v + c3 * xi - c4 * eta
            dx = dt * v + c5 * eta * dt
        else:
            v += dt / 2.0 * f / m
            dx = v * dt

        newx = x0 + dx
        newcolv = self.colvar.compute(x=newx)
        penew, fnew0 = self.potential.compute(x=newx)

        fixf = np.zeros(fnew0.shape)
        biase = 0
        for fix in self.fixes:
            fix_V, fix_f = fix.compute(x0=newx, col0=newcolv)
            biase += fix_V
            fixf += fix_f
        fnew = fnew0 + fixf

        if self.integrate == "langevin":
            v += fnew / 2.0 / m * dt
            v = c1 * v + c2 * xi
        elif self.integrate == "nve":
            v += fnew / m * dt / 2.0
        elif self.integrate == "2nd-langevin":
            v += c1 * fnew / m - c2 * v + c3 * xi - c4 * eta
        else:
            v += fnew / m * dt / 2.0

        ke = np.linalg.norm(v) ** 2 * m / 2.0
        totale = penew + ke + biase

        dE = np.abs(totale - totale0)

        self.current_dt = dt
        np.copyto(atoms.prev_positions, atoms.positions)
        np.copyto(atoms.positions, newx)
        np.copyto(atoms.colv, newcolv)

        atoms.pe, atoms.forces = self.potential.compute()

        atoms.fixf = np.zeros(atoms.forces.shape)
        atoms.biase = 0
        for fix in self.fixes:
            fix_V, fix_f = fix.compute(x0=newx, col0=newcolv)
            atoms.biase += fix_V
            atoms.fixf += fix_f

        if self.integrate == "rescale":
            count = floor(time / self.rescale_freq) + 1
            if count > self.rescale_count:
                self.rescale_count = count
                scale_factor = self.v_target / (np.linalg.norm(v))
                v *= scale_factor

        np.copyto(atoms.velocities, v)
        atoms.ke = np.sum(v ** 2) * m / 2.0
        atoms.T = atoms.ke / ndim / kB * 2.0
        atoms.totale = atoms.pe + atoms.ke + atoms.biase

        return True
