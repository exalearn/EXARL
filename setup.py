from setuptools import setup, find_packages

setup(name='exarl',
      version='0.0.1',
      description='ExaRL is a high performance reinforcement learning framework',
      install_requires=['keras', 'mpi4py', 'gym', 'ase', 'Lmfit', 'seaborn', 'scikit-learn', 'pandas', 'numba', 'pybind11', 'pytest'],
      packages=find_packages(),
      )
