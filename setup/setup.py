from setuptools import setup

setup(name='exarl',
      version='0.0.1',
      description='ExaRL is a high performance reinforcement learning framework',
      install_requires=['tensorflow-gpu>=2.0.0', 'mpi4py', 'gym==0.15.4', 'ase', 'plotille',
                        'Lmfit', 'scikit-learn', 'pandas', 'numba', 'pybind11', 'pytest',
                        'pytest-cov']
      )
