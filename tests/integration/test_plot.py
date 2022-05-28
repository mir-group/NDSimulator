from copy import deepcopy
import pytest
from ndsimulator import NDRun


class TestPlot:
    @pytest.mark.parametrize("oneplot", [True, False])
    def test_plot(self, oneplot, basic_md):
        kwargs = deepcopy(basic_md)
        kwargs["run_name"] = f"{oneplot}"
        kwargs["plot"] = True
        kwargs["oneplot"] = oneplot
        kwargs["boundary"] = [[0, 48], [0, 24]]
        run = NDRun(**kwargs)
        run.begin()
        run.run()
        run.end()
        del run
