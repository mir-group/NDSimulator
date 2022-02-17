import pytest
from ndsimulator.bias import bias_from_config


class TestBias:
    kwargs = dict(
        mtd=dict(
            mtd_w=0.01,
            mtd_sigma=[0.05, 0.05],
            mtd_dep_freq=2,
            mtd_biasf=100,
        ),
        umb=dict(
            umb_k=1,
            umb_r0=[14, 14],
        ),
    )

    @pytest.mark.parametrize("bias_name", ["mtd", "umb"])
    def test_init(self, bias_name):
        bias = bias_from_config(self.kwargs[bias_name], bias_name)

    # @pytest.mark.parametrize("bias_name", ["mtd", "umb"])
    # def test_run(self, bias_name, basic_md):
    #     kwargs = deepcopy(self.kwargs[bias_name])
    #     kwargs.update(basic_md)
    #     kwargs.update(run_name="test_run")
    #     run = NDRun(**kwargs)
    #     run.begin()
    #     run.run()
    #     run.end()
    #     del run
