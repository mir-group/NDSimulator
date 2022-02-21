#!/usr/bin/env python
import numpy as np
from ndsimulator.data import AllData


class Stat(AllData):
    def __init__(
        self,
        track: dict = {"p": False, "v": False, "f": False},
        freq: int = 1,
        mem: int = 100000,
    ):
        AllData.__init__(self)
        self.freq = freq
        self.mem = mem
        self.T = []
        self.pe = []
        self.ke = []
        self.totale = []
        self.biase = []
        self.time = []
        self.dt = []
        self.positions = []
        self.velocities = []
        self.forces = []
        self.colv = []
        self.track_v = track["v"]
        self.track_p = track["p"]
        self.track_f = track["f"]

    def track_everything(self, option: bool = True):
        self.track_v = option
        self.track_p = option
        self.track_f = option

    def initialize(self, run):
        AllData.__init__(self, run=run)
        if self.plot and self.plot.oneplot:
            self.track_p = True

    def modify(self):
        atoms = self.atoms
        if self.pe:
            self.pe[-1] = atoms.pe
            self.T[-1] = atoms.T
            self.ke[-1] = atoms.ke
            self.totale[-1] = atoms.totale
            self.biase[-1] = atoms.biase
            if self.track_v:
                self.velocities[-1] = np.copy(atoms.velocities)
            if self.track_p:
                self.positions[-1] = np.copy(atoms.positions)
                self.colv[-1] = np.copy(atoms.colv)
            if self.track_f:
                self.forces[-1] = np.copy(atoms.forces)

    def append(self, time, dt):
        if len(self.dt) > self.mem:
            self.clean()
        self.time.append(time)
        self.dt.append(dt)
        atoms = self.atoms
        self.pe.append(atoms.pe)
        self.T.append(atoms.T)
        self.ke.append(atoms.ke)
        self.totale.append(atoms.totale)
        self.biase.append(atoms.biase)
        if self.track_v:
            self.velocities.append(np.copy(atoms.velocities))
        if self.track_p:
            self.positions.append(np.copy(atoms.positions))
            self.colv.append(np.copy(atoms.colv))
        if self.track_f:
            self.forces.append(np.copy(atoms.forces))

    def clean(self):
        m = int(self.mem * 0.9)
        del self.time[:m]
        del self.pe[:m]
        del self.T[:m]
        del self.ke[:m]
        del self.totale[:m]
        del self.biase[:m]
        if self.track_v:
            del self.velocities[:m]
        if self.track_p:
            del self.positions[:m]
            del self.colv[:m]
        if self.track_f:
            del self.forces[:m]

    def reverse(self):
        self.time = -self.time.reverse()
        self.dt.reverse()
        self.pe.reverse()
        self.T.reverse()
        self.ke.reverse()
        self.totale.reverse()
        self.biase.reverse()
        if self.track_v:
            self.velocitties = -self.velocities.reverse()
        if self.track_p:
            self.positions.reverse()
            self.colv.reverse()
        if self.track_f:
            self.forces.reverse()
