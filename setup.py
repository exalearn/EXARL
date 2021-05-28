from setuptools import setup

setup(name='exarl',
      version='0.0.1',
      description='ExaRL is a high performance reinforcement learning framework',
      install_requires=['horovod', 'tensorflow-gpu>=2.0.0', 'mpi4py', 'gym', 'ase',
                        'Lmfit', 'scikit-learn', 'pandas', 'numba', 'pybind11', 'pytest']
      )
