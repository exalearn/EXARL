# ExaRL Spack Repository

To install EXARL via Spack, take the following steps:

## Install Spack

The `setup-env.sh` script in the spack repository configures your environment
for development with spack.

```console
$ git clone https://github.com/spack/spack.git
$ cd spack
$ git checkout v0.16.1
$ source ./spack/share/spack/setup-env.sh
```

## Install EXARL

Then, after cloning the repository, add the `spack` subdirectory of EXARL as a
spack repository which will provide the `py-exarl` spack package.

```console
$ git clone https://github.com/exalearn/EXARL.git EXARL
$ cd EXARL
$ spack repo add ./spack
```

At this point, you should be able to view the spack package with `spack info`
like so:

```console
$ spack info py-exarl
PythonPackage:   py-exarl

Description:
    A scalable software framework for reinforcement learning environments
    and agents/policies used for the Design and Control applications

Homepage: https://github.com/exalearn/EXARL
...
$ spack install py-exarl
...
```

## Installing on Summit

On Summit, you likely want to re-use the IBM WML module for most of the
machine learning libraries. In this case, you may use the environment file
provided in this repository:

```console
$ # Load modules and clone IBM ml module (this is laid out in the EXARL wiki)
$ module load spectrum-mpi/10.3.1.2-20200121
$ module load gcc/6.4.0
$ module load ibm-wml-ce/1.7.0-3
$ module load cuda/10.1.243
$ conda create --name exarl --clone /sw/summit/ibm-wml-ce/anaconda-base/envs/ibm-wml-ce-1.7.0-3

$ # Install mpi4py into conda installation
$ # Note that you could also just let spack install this for you, however I chose
$ # to install this particular dependency by-hand at the time of writing.
$ wget https://github.com/mpi4py/mpi4py/archive/3.0.3.tar.gz
$ tar -xvzf 3.0.3.tar.gz
$ python setup.py build --mpicc=`which mpicc`
$ python setup.py install --prefix=$CONDA_PREFIX

# Install EXARL via spack package
$ cd EXARL
$ spack repo add ./spack
$ spack env create exarl ./spack/environments/summit/exarl-conda.yaml
$ spack env activate exarl
$ spack install
```

If you see spack installing any packages that you know are already provided by
a module or you have already installed locally, you may add them to the
environment file via `spack config edit`.

The key benefit of using this environment file is that most packages already
avialable on Summit are discoverable by spack. For example, OpenJDK does not
build on Power 9 at the time of this writing, which means you will not be able
to install Bazel, which means you won't be able to install Tensorflow. Many
of these difficulties are resolved by using what's already installed on the
system.

Note that this spack environment by default will install the `develop` branch
of EXARL, ***not the branch you cloned from github in your current working
directory***. You may change this by adding a version to the spack package (
in `EXARL/spack/packages/py-exarl/package.py`) for
whatever branch you're working on and changing the spack environment with
`spack config edit` and replacing the `exarl.py-exaxrl@develop` with
`exarl.py-exarl@your-branch-here`.

## Installing on Ascent

Installing on Ascent is almost identical to installing on Summit:

```console
$ # Load modules and clone IBM ml module (this is laid out in the EXARL wiki)
$ module load spectrum-mpi/10.3.1.2-20200121
$ module load gcc/6.4.0
$ module load ibm-wml-ce/1.7.0-0
$ module load cuda/10.1.243

$ cd EXARL
$ spack env create exarl ./spack/environments/ascent/exarl-conda.yaml
$ spack env activate exarl
$ spack install
```

Note that the spack environments look in your anaconda environment for many of
the python packages, since the more complicated packages are already installed
in the IBM module. If you would like to use an alternate installation for any
python package in particular, just specify the location or module of that
package in the environment file.
