from ndsimulator.data import AllData


class Potential(AllData):
    require_colvar = False
    ndim = -1

    def compute(self, x):
        raise NotImplementedError
