import pytest
import tempfile


@pytest.fixture(scope="function")
def tempdir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture(scope="function")
def basic_md(tempdir):
    _input = dict(
        root=tempdir,
        method="md",
        ndim=2,
        potential="Mueller2d",
        steps=10,
        colvar_name="Original",
        true_colvar_name="Original",
        integrate="rescale",
        rescale_freq=3,
        dt=1.0,
        temperature=300,
        mass=1.0,
        x0=[12.0, 12.0],
        plot=False,
        movie=False,
        dump=True,
        dump_freq=1,
        boundary=[[2, 40], [2, 40]],
        verbose="debug",
    )
    yield _input


@pytest.fixture(scope="function")
def basic_mc(tempdir):
    _input = dict(
        root=tempdir,
        method="mc",
        ndim=2,
        potential="Mueller2d",
        steps=10,
        colvar_name="Original",
        true_colvar_name="Original",
        integrate="rescale",
        rescale_freq=3,
        dt=1.0,
        temperature=300,
        mass=1.0,
        x0=[12.0, 12.0],
        boundary=[[2, 40], [2, 40]],
        plot=False,
        movie=False,
        dump=True,
        dump_freq=1,
        verbose="debug",
        mc_bounds=[1, 1],
        mc_global_bounds=[50, 50],
    )
    yield _input
