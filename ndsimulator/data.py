"""
initialize() function will be called by ndrun at the
initialization phase. one can also write customized __init__() and
initialize() functions. Just make sure that the parent class
function is explicitely called at the beginning of the customized
function as well.
"""

import numpy as np
from ndsimulator.constant import kB, au_mass


class AllData:
    def __init__(self, run=None):
        if run:
            self.atoms = run.atoms
            self.potential = run.potential
            self.output = run.output
            self.stat = run.stat
            self.fixes = run.fixes
            self.engine = run.engine
            self.plot = run.plot
            self.dump = run.dump
            self.colvar = run.colvar
            self.true_colvar = run.true_colvar
            self.random = run.random
            self.kBT = kB * run.temperature
            self.amass = au_mass * run.mass
            self.m = self.amass
            self.temperature = run.temperature
            self.ndim = run.ndim
            self.steps = run.steps
            self.screen = run.screen
            self.x0 = run.x0
        else:
            self.atoms = None
            self.output = None
            self.potential = None
            self.stat = None
            self.fixes = None
            self.engine = None
            self.plot = None
            self.dump = None
            self.colvar = None
            self.true_colvar = None
            self.random = None

    def initialize(self, pointer):
        AllData.__init__(self, pointer)


class Atom(AllData):
    def __init__(self, boundary=[[0.0, 48.0], [0.0, 40.0]], run=None):
        self.boundary = np.array(boundary)
        super(Atom, self).__init__(run)

    def initialize(self, run):
        AllData.__init__(self, run)

        self.colv = None
        self.positions = None
        self.prev_positions = None
        self.prev_colv = None
        self.velocities = None
        self.forces = None
        self.bforces = None
        self.T = None

        if type(self.engine).__name__ != "Read_Dump":
            if not isinstance(run.x0, str):
                self.positions = np.copy(run.x0)
            else:
                self.positions = self.random.rand(self.ndim)
            self.prev_positions = np.array(self.positions)
            self.velocities = np.zeros(self.ndim)
            self.forces = np.zeros(self.ndim)

        self.pe = 0
        self.ke = 0
        self.biase = None
        self.totale = 0
