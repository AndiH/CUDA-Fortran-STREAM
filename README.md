# CUDA Fortran STREAM Benchmark

Variant of the STREAM benchmark written in CUDA Fortran (and, hence, working on the GPU).

The [PGI compiler](https://www.pgroup.com/resources/cudafortran.htm) is required to compile.

The four STREAM benchmarks are run on the GPU, that is (with arrays `a`, `b`, `c`):

* COPY: `c_j = a_j`
* SCALE: `b_j = scalar * c_j`
* ADD: `c_j = a_j + b_j`
* TRIAD: `a_j = b_j + scalar * c_j`

For information is available in `stream.F90`. The `Makefile` should suffice to call `make run` for compiling and running the four micro-benchmarks.


See also [www.streambench.org](http://www.streambench.org), especially for notes on publishing results based no (variants of) the STREAM benchmark.
