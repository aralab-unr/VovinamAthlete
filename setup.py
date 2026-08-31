"""Installation script for the 'vovinamathlete_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
]

# Installation operation
setup(
    name="vovinamathlete_mjlab",
    packages=["vovinamathlete_mjlab"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
