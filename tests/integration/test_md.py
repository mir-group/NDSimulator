import pytest
from pyfile_utils import Config
from ndsimulator.ndrun import NDRun


class TestMD:
    @pytest.mark.parametrize(
        "integrate", ["rescale", "langevin", "2nd-langevin", "nve"]
    )
    def test_simple_md(self, tempdir, integrate):
        config = Config.from_file("examples/2d-md.yaml")
        config.root = tempdir
        config.run_name = integrate
        config.steps = config.steps // 100
        config.integration = integrate
        run = NDRun(**dict(config))
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
