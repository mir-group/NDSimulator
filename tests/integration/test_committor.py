from ndsimulator.ndrun import NDRun
from pyfile_utils import Config


class TestCommittor:
    def test_simple_committor(self, tempdir):
        config = Config.from_file("examples/2d-committor.yaml")
        config.root = tempdir
        config.steps = config.steps // 100
        run = NDRun(**dict(config))
        run.begin()
        run.run()
        run.end()
        del run
