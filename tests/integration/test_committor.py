from copy import deepcopy
from ndsimulator.ndrun import NDRun


class TestCommittor:
    def test_simple_committor(self, basic_md):
        kwargs = deepcopy(basic_md)
        kwargs["run_name"] = "committor"
        kwargs["method"] = "committor"
        run = NDRun(**kwargs)
        run.begin()
        run.run()
        run.end()
        del run
