from copy import deepcopy
from ndsimulator import NDRun


class TestMinimize:
    def test_min(self, basic_md):
        kwargs = deepcopy(basic_md)
        kwargs["method"] = "minimize"
        kwargs["run_name"] = "minimize"
        run = NDRun(**kwargs)
        run.begin()
        run.run()
        run.end()
        del run
