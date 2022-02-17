import pytest
from copy import deepcopy
from ndsimulator.ndrun import NDRun


class TestMD:
    @pytest.mark.parametrize(
        "integrate", ["rescale", "langevin", "2nd-langevin", "nve"]
    )
    def test_rescale(self, integrate, basic_md):
        kwargs = deepcopy(basic_md)
        kwargs["run_name"] = integrate
        kwargs["integrate"] = integrate
        run = NDRun(**kwargs)
        run.begin()
        run.run()
        run.end()
        del run

    # def test_changedt(self):
    #     arguments = copy.deepcopy(self._input)
    #     arguments['integrate'] = "nve"
    #     arguments['md_fixdt'] = False
    #     run = NDRun(**arguments)
    #     run.begin()
    #     run.run()
    #     run.end()
    #     del run
