DISCONTINUATION OF PROJECT.

This project will no longer be maintained by Intel.

Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project. 

Intel no longer accepts patches to this project.

If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project. 
HFAV
====

High-performance Fusion And Vectorization (formerly "Rolling Thunder")

Overview
--------

This is a prototype that demonstrates how certain code transformation techniques may be automatically applied to a suitable input; in particular, it aims to automatically fuse and vectorize kernels while minimizing intermediate storage. For computations where *pure* kernels are applied to regular grids, particularly where kernels pass information to on another, hfav may provide speedup.

hfav accepts a declarative input file that specifies the function prototype for each kernel along with information about each parameter and the iteration space that the kernel should be applied to. Terminal conditions (*axioms* and *goals*) are supplied, along with options about code generation and output. The resulting output is indended to be linked into the original code, perferably in a fashion that enables inlining (which is necessary for auto-vectorization).

License
-------

This software and all but one example is distributed with a modified Apache License 2.0. See LICENSE for details; the modification is an exception that code generated with this software is only subject to the limited liability clauses of the Apache 2.0 license (in particular, we don't retain copyright on generated code).

The hydro2d example is subject to the CeCILL license; see examples/hydro2d/LICENSE for details.

Usage
-----

hfav.py is the top-level interface to hfav. It is invoked as:

    hfav.py [-h] [-d] [-o OUTPUT_LOCATION] [-s STORAGE] [-v {0,1,2}] FILE

### Options

- `FILE`: input YAML file (*mandatory*)
- `-h, --help`: show help message and exit
- `-d, --debug`: enable debug output
- `-o OUTPUT_LOCATION, --output OUTPUT_LOCATION`: override output location; "-" gives stdout
- `-s STORAGE, --storage STORAGE`: where to place temporary arrays (default: stack)
- `-v {0,1,2}, --verbosity {0,1,2}` level of verbosity while processing

It can be useful to export the environment variable `HFAVROOT` to the `hfav/` directory contained in this source distribution.

Examples
--------

The YAML format accepted by hfav is best understood by looking at examples. See the `examples/` directory for more detail.

The `hydro2d/` directory contains a more comprehensive example complete with Makefile integration.

More information
----------------

A paper on the ideas behind HFAV will be presented at the Seventh Internation Workshop on Domain-Specific Languages and High-Level Framworks for High Performance Computing (WOLFHPC) at ACM/IEEE Supercomputing in Denver in November 2017.

    Jason D. Sewall and Simon J. Pennycook. 2017. High-Performance Code Generation though Fusion and Vectorization. To be presented at WOLFHPC 2017, Denver. Nobember 2017.

A preprint is available at arXiv: [https://arxiv.org/abs/1710.08774](https://arxiv.org/abs/1710.08774).

Contributors
------------

- John Pennycook (john.pennycook@intel.com)
- Jason Sewall (jason.sewall@intel.com)
