# Mochi S4M (Share for Me)

This service provides a simple non-blocking broadcast/receive
mechanism based on Mochi.

## Installing

Make sure you have [spack](https://spack.io/) installed and setup.
If needed, install it and set it up as follows:

```
$ git clone https://github.com/spack/spack.git
$ . spack/share/spack/setup-env.sh
```

You then need to clone the `mochi-spack-packages` repository
and make it available to spack:

```
$ git clone https://github.com/mochi-hpc/mochi-spack-packages.git
$ spack repo add mochi-spack-packages
```

Finally, you can install S4M as follows:

```
$ spack install py-mochi-s4m
```

## Using

S4M has a very simple API consisting of an `S4MService` class with
two functions: `broadcast`, and `receive`. It requires mpi4py to
bootstrap the set of processes. The [test.py](test/test.py) file
provides a comprehensive use case.
