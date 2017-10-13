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

Contributors
------------

- John Pennycook (john.pennycook@intel.com)
- Jason Sewall (jason.sewall@intel.com)
