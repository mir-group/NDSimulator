import numpy as np
import pytest

from ndsimulator.potentials import (
    DoubleWell,
    ThreeHole2d,
    Mueller2d,
    Gaussian,
    Gaussian2d,
    potential_from_config,
)

list_of_potentials = pytest.mark.parametrize(
    "potential",
    [DoubleWell, ThreeHole2d, Mueller2d, Gaussian, Gaussian2d],
)


class TestInstantiation:
    @list_of_potentials
    def test_init(self, potential):
        ins = potential()

    @list_of_potentials
    def test_compute(self, potential):
        ins = potential()
        if not ins.require_colvar:
            if ins.ndim is None:
                x = np.random.random(10)
            else:
                x = np.random.random(ins.ndim)
            ins.compute(x)


@list_of_potentials
def test_build(potential):
    ins = potential_from_config({"potential": potential.__name__})
    print(potential)
    print(type(ins).__name__)
    assert isinstance(ins, potential)
