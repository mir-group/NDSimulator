import numpy as np
from ndsimulator.data import AllData
from ndsimulator.constant import kB


class PEBias(AllData):
    """The PE_bias object implements bias that based on potential energy

    It penalize all the configuration that has a potential energy lower than a thredshold

    """

    def __init__(
        self,
        thred=0.0,
        run=None,
    ):
        self.thred = thred
        super(PEBias, self).__init__(run)

    def initialize(self, pointer):

        AllData.__init__(self, run=pointer)

        self.VR = 0

    def update(self, step, time):
        pass

    def compute(self, x0=None, col0=None):

        # TO DO check the force form
        if x0 is None:
            x = self.atoms.positions
            pe = self.atoms.pe
            f = self.atoms.forces
        else:
            x = x0
            pe, f = self.potential.compute(x)

        if pe < self.thred:
            f = -np.array(f)
            V = self.thred - pe
        else:
            f = np.zeros(self.ndim)
            V = 0

        if x0 is None:
            self.VR = V

        return V, f

    def force(self, x0=None, col0=None):
        # TO DO check the force form
        if x0 is None:
            x = self.atoms.x
            pe = self.atoms.pe
            f = self.atoms.forces
        else:
            x = x0
            pe, f = self.potential.compute(x)

        if pe < self.thred:
            return -np.array(f)
        else:
            return np.zeros(self.ndim)

    def energy(self, x0=None, col0=None):

        if x0 is None:
            x = self.atoms.positions
            pe = self.atoms.pe
        else:
            x = x0
            pe, f = self.potential.compute(x)

        if pe < self.thred:
            V = self.thred - pe
        else:
            V = 0

        if x0 is None:
            self.VR = V
        return V

    def dump_data(self):
        line = ""
        return line

    def projection(self, X, Y):
        pass
