from ndsimulator.data import AllData


class Colvar(AllData):
    """The parent class of all the collective variables"""

    colvardim = None

    def __init__(self, dim=2, run=None):
        self.dim = dim
        super(Colvar, self).__init__(run)

    def initialize(self, pointer):
        AllData.initialize(self, pointer)
        self.ndim = pointer.ndim
        if self.colvardim is not None:
            if self.colvardim != self.dim:
                raise NameError(
                    "dimension of collective variable {} does not match with the function {}".format(
                        self.dim, self.colvardim
                    )
                )

    def compute(self, x):
        raise NotImplementedError

    def jacobian(self, x):
        raise NotImplementedError
