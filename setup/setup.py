from setuptools import setup, find_packages

setup(
    name='exarl',
    version='0.0.1',
    description='ExaRL is a high performance reinforcement learning framework',
    package_dir={'': '..'},
    packages=find_packages(where='..', include=['exarl', 'exarl.*']),
    install_requires=[
        'tensorflow>=2.0.0',
        'mpi4py',
        'gym<0.26.0',
        'ase',
        'plotille',
        'Lmfit',
        'scikit-learn',
        'pandas',
        'numba',
        'pybind11',
        'pytest'
    ],
    python_requires='>=3.7',
)