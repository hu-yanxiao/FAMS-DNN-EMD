# setup.py
from setuptools import setup, find_packages
import shutil
shutil.move("fams_dnn", "ptagnn")
setup(
    name="ptagnn",
    version='1.0',
    packages=find_packages(),
    author='wqzhang-group',
    description='MLIP',
)