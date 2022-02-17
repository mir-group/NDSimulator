import pytest
from ndsimulator.colvars import Proj5dto2d, XmY, colvar_from_config


list_of_colvars = pytest.mark.parametrize("colvar", [Proj5dto2d, XmY])


class TestInstantiation:
    @list_of_colvars
    def test_init(self, colvar):
        ins = colvar()

    @list_of_colvars
    def test_compute(self, colvar):
        ins = colvar()


@list_of_colvars
def test_build(colvar):
    colvar_from_config({"colvar_name": colvar.__name__})
