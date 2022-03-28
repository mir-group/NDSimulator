from setuptools import setup, find_packages
from pathlib import Path

package_names = ["ndsimulator"]
_name = "_".join(package_names)
name = "-".join(package_names)

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / _name / "_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name=name,
    version=f"{version}",
    author="Lixin Sun",
    python_requires=">=3.6.9",
    packages=find_packages(include=[name, _name, _name + ".*"]),
    install_requires=[
        "numpy",
        "pyfile-utils>=1.0.4",
    ],
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "ndsimulator = ndsimulator.scripts.run:main",
            "ndPEL = ndsimulator.scripts.plot_pot:main",
        ]
    },
    zip_safe=True,
)
