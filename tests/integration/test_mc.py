from copy import deepcopy
from ndsimulator import NDRun


class TestMC:
    def test_wholemdprocess(self, basic_mc):
        kwargs = deepcopy(basic_mc)
        kwargs["run_name"] = "mc"
        run = NDRun(**kwargs)
        run.begin()
        run.run()
        run.end()
        del run

    def test_nve(self, basic_mc):
        kwargs = deepcopy(basic_mc)
        kwargs["run_name"] = "mc_nve"
        kwargs["propose_dim"] = "all"
        run = NDRun(**kwargs)
        run.begin()
        run.run()
        run.end()
        del run
