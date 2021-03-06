import numpy as np

from .colvar import Colvar


class Original(Colvar):
    def __init__(self, dim=2, run=None):
        self.dim = dim
        Original.colvardim = dim
        super(Colvar, self).__init__(run)

    def compute(self, x):
        return np.copy(x)

    def jacobian(self, x):
        return np.identity(len(x.flat))


class Map50to2d(Colvar):
    colvardim = 2
    vec1 = np.array(
        [
            0.9621493306867017,
            0.2611502170144795,
            0.0000000000000000,
            0.7331647024340944,
            0.8708183789325146,
            0.0805017856037096,
            0.0000000000000000,
            0.5785249542521097,
            0.3783625809001937,
            0.0000000000000000,
            0.0571747072559683,
            0.9724187143800709,
            0.0000000000000000,
            0.041131319898087,
            0.0000000000000000,
            0.2765931554549067,
            0.0912801608664523,
            0.0000000000000000,
            0.0431413377076534,
            0.21668041341737354,
            0.04381849659171735,
            0.0000000000000000,
            0.0120163151455723,
            0.00000000000000000,
            0.0038527070151484,
            0.729285346078329,
            0.09400213558501833,
            0.00000000000000000,
            0.0345084258996749,
            0.0000000000000000,
            0.0602460013004428,
            0.0000000000000000,
            0.0102889267052817,
            0.7013536441873398,
            0.00125254636835465,
            0.000000000000000,
            0.0006896724138744,
            0.0042907976312725,
            0.0100253907332234,
            0.0005460275194114628,
            0.00567411387523127,
            0.0025103010306175,
            0.0007127704701611,
            0.0001890413255533,
            0.0407958220608383,
            0.0061146053157138,
            0.00491691153887918,
            0.0024312731237804,
            0.0761174326669661,
            0.00586435669539784,
        ]
    )
    vec2 = np.array(
        [
            0.7200723641543899,
            0.938701822380177,
            0.063114103710782,
            0.0,
            0.5650229051546,
            0.0,
            0.0,
            0.0,
            0.0047707786427023,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0254131316679295,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0006544091825368,
            0.0,
            0.0,
            0.7915907738260504,
            0.0006566689112959,
            0.0308050889004108,
            0.0306030445075646,
            0.0683335335391172,
            0.0927967809881871,
            0.0,
            0.0,
            0.0668273272540637,
            0.9543323517079718,
            0.0,
            0.0,
            0.0,
            0.0705778822165263,
            0.0,
            0.0167903655654037,
            0.9330919923191575,
            0.0,
            0.9176521018708412,
            0.0505357766611562,
            0.0,
            0.0750136062509377,
            0.5709571823782142,
            0.0591655909593535,
            0.0,
            0.0,
            0.0113295431324928,
            0.0,
            0.0444879007946433,
        ]
    )
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    Kmatrix1 = np.diag(vec1)
    Kmatrix2 = np.diag(vec2)
    Kmatrix1[0, 1] = 1.0
    Kmatrix1[1, 0] = 1.0
    Kmatrix2[3, 5] = 0.2
    Kmatrix2[5, 3] = 0.2
    Kmatrix2[4, 8] = 0.5
    Kmatrix2[8, 4] = 0.5
    Kmatrix2[2, 4] = 1.0
    Kmatrix2[4, 2] = 1.0

    def compute(self, x):
        tx = x.reshape([1, self.ndim])
        newx = np.hstack(
            [(tx.dot(self.Kmatrix1)).dot(tx.T), (tx.dot(self.Kmatrix2)).dot(tx.T)]
        ).reshape(
            [
                2,
            ]
        )
        return newx

    def jacobian(self, x):
        tx = x.reshape([1, self.ndim])
        jacob = 2 * np.vstack([tx.dot(self.Kmatrix1), tx.dot(self.Kmatrix2)])
        return jacob
