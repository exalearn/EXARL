# ExaRL Spack Repository

This repository contains the spack package for ExaRL, as well as environments
to build ExaRL on target platforms.

The directories in this repository with a brief description:

- `config`: Contains configuration files for target platforms
- `environments`: Contains YAML spack environment files to instantiate ExaRL
    and its dependencies.
- `packages`: Contains spack packages for ExaRL and any other dependencies that
    may have to be patched for ExaRL to build/run correctly.

Each of these directories will be discussed in greater detail below.

## Config

The config directory contains spack configuration files which tell your spack
installation where many preinstalled packages exist. For example, we likely
prefer the environment modules for the compilers we use, rather than building
any compilers we use from scratch.

On Ascent for example:

```console
$ # Assuming you already have spack activated:
$ cp ./config/ascent/* $SPACK_ROOT/etc/spack/
```

## Environments

Each directory under this directory contains environments which instantiate the
ExaRL stack on a given platform.

Example usage:
```console
$ ssh login1.ascent.olcf.ornl.gov
$ git clone https://github.com/spack/spack.git
$ source ./spack/share/spack/setup-env.sh
$ git clone https://github.com/exalearn/ExaRL.git
$ spack repo add ./ExaRL/.spack
$ spack env create ascent-exarl ./ExaRL/.spack/environments/ascent/exarl.yaml
$ spack install
```

## Packages

Currently, the only package in this directory is for ExaRL. Should this
repository be made public, this package may be upstreamed.
