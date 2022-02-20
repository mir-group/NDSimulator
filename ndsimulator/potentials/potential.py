from ndsimulator.data import AllData


class Potential(AllData):
    require_colvar = False
    ndim = -1

    def __init__(self, run=None):
        super(Potential, self).__init__(run=run)

    def initialize(self, pointer):
        assert pointer.ndim == self.ndim
        AllData.initialize(self, pointer)

    def compute(self, x):
        raise NotImplementedError
