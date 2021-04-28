# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack import *
import os


class PyExarl(PythonPackage):
    """A scalable software framework for reinforcement learning environments
    and agents/policies used for the Design and Control applications"""

    homepage  = 'https://github.com/exalearn/EXARL'
    git = 'https://github.com/exalearn/EXARL.git'

    version('master', branch='master')
    version('develop', branch='develop')
    version('0.6.0', tag='devel_v0.6')
    version('0.5.0', tag='devel_v0.5')
    version('0.4.0', tag='devel_v0.4')
    version('0.4.0', tag='devel_v0.4')
    version('0.3.0', tag='devel_v0.3')
    version('0.2.0', tag='devel_v0.2')
    version('0.1.0', tag='devel_v0.1')

    depends_on('python@3.6:',   type=('build', 'run'))
    depends_on('git-lfs',       type=('build'))
    depends_on('py-setuptools', type=('build'))

    depends_on('py-ase',          type=('build', 'run'))
    depends_on('py-lmfit',        type=('build', 'run'))
    depends_on('py-seaborn',      type=('build', 'run'))
    depends_on('py-keras',        type=('build', 'run'))
    depends_on('py-mpi4py',       type=('build', 'run'))
    depends_on('py-scikit-learn', type=('build', 'run'))
    depends_on('py-tensorflow',   type=('build', 'run'))
    depends_on('py-torch',        type=('build', 'run'))
    depends_on('py-torchvision',  type=('build', 'run'))

    phases = ['install']
