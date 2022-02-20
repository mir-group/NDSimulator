from ndsimulator.data import AllData


class Colvar(AllData):
    """The parent class of all the collective variables"""

    colvardim = None

    def __init__(self, run=None):
        super(Colvar, self).__init__(run)

    def initialize(self, pointer):
        AllData.initialize(self, pointer)
        self.ndim = pointer.ndim

    def compute(self, x):
        raise NotImplementedError

    def jacobian(self, x):
        raise NotImplementedError
