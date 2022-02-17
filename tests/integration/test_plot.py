from copy import deepcopy
import pytest
from ndsimulator import NDRun


class TestPlot:
    @pytest.mark.parametrize("oneplot", [True, False])
    @pytest.mark.parametrize("light_plot", [True, False])
    def test_plot(self, oneplot, light_plot, basic_md):
        kwargs = deepcopy(basic_md)
        kwargs["run_name"] = f"{oneplot}_{light_plot}"
        kwargs["plot"] = True
        kwargs["light_plot"] = light_plot
        kwargs["oneplot"] = oneplot
        kwargs["boundary"] = [[0, 48], [0, 24]]
        run = NDRun(**kwargs)
        run.begin()
        run.run()
        run.end()
        del run
