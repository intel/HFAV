#!/usr/bin/python

# hfav.py; top-level invocation and parsing

# Copyright 2017 Intel Corporation
#
# GENERATED CODE EXEMPTION
#
# The output of this tool does not automatically import the Apache
# 2.0 license, except the output will continue to be subject to the
# limitation of liability clause in the Apache 2.0 license. Users may
# license their output under any license they choose but the liability
# of the authors of the tool for that output is governed by the
# limitation of liability clause in the Apache 2.0 license.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import re
import os
import argparse
import yaml
from hfav.infer import dag_chain, rule_arg, rule, axiom, goal, rule_group, ivar_axiom, codeblock, reduction_op, reduction_initializer, reduction_finalizer
from hfav.ispace import iteration_space
from hfav.term import symbolic_constant
from hfav.analyze import simple_generator, rolling_generator, rap_dual
from hfav.c99 import codegen, c99_generator
from hfav.cpp import cpp_generator
from hfav.inest import inest_dag
from hfav import parse

def parse_declaration(declaration):
    m = re.match(r"(?:(\w+)[\s]+)?(\w+)\(([\w\d\s,&]*)\);", declaration)
    if m is None:
        raise SyntaxError("Malformed kernel declaration: %s" % (declaration))  # TODO: More information...
    else:
        rtype = m.group(1)
        kname = m.group(2)
        types = {}
        positions = {}
        try:
            varlist = m.group(3).replace("&", "").replace("*", "")
            pos = 0
            for v in re.split(r",\s*", varlist):
                vsplit = re.split(r"\s*", v)
                if len(vsplit) != 2:
                    raise SyntaxError("Can't tell if %s is a type or a variable name." % (v))
                vtype = vsplit[0]
                vname = vsplit[1]
                types[vname] = vtype
                positions[vname] = pos
                pos = pos + 1
        except IndexError:
            pass
        if rtype is not None and rtype != 'void':
            types['<return>'] = rtype
            positions['<return>'] = -1
        return (kname, types, positions)

# top level list
# [ iter level
#   itspace
# ]

# loop [prologue, steady, epilogue] ident

# ident parent, children (loops, raps)

# loop
#

def hfav_run_yaml():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="YAML front-end for High-performance Inference Fusion Into Vectorization (hfav)")
    parser.add_argument('-d', '--debug', dest='debug_output', action='store_true', default=False, help='enable debug output')
    parser.add_argument('-o', '--output', dest='output_location', action='store', default=False, help='override output location "-" gives stdout')
    parser.add_argument('-s', '--storage', dest='storage', action='store', default='stack', help='where to place temporary arrays (default: stack)')
    parser.add_argument('-v', '--verbosity', dest='verbosity', choices=['0', '1', '2'], action='store', default=0, help='verbosity level')
    parser.add_argument('FILE', help='Input YAML file')
    args = parser.parse_args()
    debug_output = args.debug_output
    extra_output = args.verbosity
    filename = args.FILE
    storage = args.storage

    logging.info("Loading input file %s", filename)
    config = yaml.load(file(filename, 'r'))
    kernels = []
    axioms = []
    goals = []

    if os.environ.get('HFAVROOT') is None:
        logging.warning("Please set HFAVROOT environment variable to your hfav directory...\n")
        hfavroot = "hfav"
    else:
        hfavroot = os.environ.get('HFAVROOT')

    # Read kernels
    for kname, kparams in config["kernels"].items():

        name, vtype, vpos = parse_declaration(kparams["declaration"])
        vrule = {}

        iargs = []
        if "inputs" in kparams.keys():
            for line in kparams["inputs"].splitlines():
                input_li = line.partition(":")
                vname = input_li[0].strip()
                if vname == '<return>':
                    raise SyntaxError("<return> cannot be used as an input! (%s)" % (name))
                vrule[vname] = input_li[2].strip()
                iargs.append(rule_arg(vpos[vname], vtype[vname], vrule[vname], "input"))

        oargs = []
        got_return = False
        if "outputs" in kparams.keys():
            for line in kparams["outputs"].splitlines():
                output = line.partition(":")
                vname = output[0].strip()
                if vname == '<return>':
                    if got_return:
                        raise SyntaxError("Got multiple <return>s! (%s, %s)" % (vrule[vname], output[2].strip()))
                    got_return = True
                if vname in vrule.keys():
                    logging.warning("Parameter \"%s\" specified as an input and output to kernel \"%s\" -- here be dragons...", vname, kname)

                m = re.match(r"reduction\((.+):(.+)\)", output[2].strip())
                if m is not None:
                    opkey = m.group(1)
                    if opkey not in reduction_op.supported().keys():
                        logging.error("%s is not a recognized reduction, must be one of: %s", opkey, map(str, reduction_op.supported().keys()))
                    red_op = reduction_op.supported()[opkey]
                    vrule[vname] = "_reduction(%s)" % m.group(2)
                    iargs.append(rule_arg(vpos[vname], vtype[vname], "_init(%s)" % m.group(2), "input"))
                    kernels.append(reduction_initializer(parse.parser(m.group(2)).expr(), red_op, vtype[vname]))
                    kernels.append(reduction_finalizer(parse.parser(m.group(2)).expr(), red_op, vtype[vname]))
                else:
                    vrule[vname] = output[2].strip()

                oargs.append(rule_arg(vpos[vname], vtype[vname], vrule[vname], "output"))

        for vname in vtype.keys():
            if vname not in vrule:
                logging.warning("No replacement rule for parameter \"%s\" passed to kernel \"%s\" was specified -- assuming a global input of the same name exists.", vname, kname)
                vrule[vname] = "%s" % (vname)
                iargs.append(rule_arg(vpos[vname], vtype[vname], vrule[vname], "input"))
                axioms.append(axiom.read(vrule[vname], vrule[vname], vtype[vname]))

        kernel = rule.read(name, iargs, oargs)
        kernels.append(kernel)

    # Read code blocks
    code_blocks = []
    try:
        cbs = config["code blocks"].items()
        for name, cb in cbs:
            block = codeblock.read(name, cb)
            code_blocks.append(block)
    except KeyError:
        logging.warning("No code blocks specified; assuming no boundary conditions -- \"to infinity and beyond!\"")

    # Read inputs
    if "inputs" in config["globals"]:
        for line in config["globals"]["inputs"].splitlines():
            input_li = line.partition("=>")
            if input_li[1] == "=>":
                decl = re.split(r"\s*", input_li[0].strip(), 1)
                axioms.append(axiom.read(decl[1], input_li[2].strip(), decl[0]))
            else:
                decl = re.split(r"\s*", input_li[0].strip(), 1)
                # this case makes sense, where the input is implicitly the same as the output
                axioms.append(axiom.read(decl[1], decl[1], decl[0]))
    else:
        logging.warning("No global inputs specified -- things are unlikely to work except in pathological cases.")

    # Read outputs
    if "outputs" in config["globals"]:
        for line in config["globals"]["outputs"].splitlines():
            output = line.partition("=>")
            if output[1] == "=>":
                decl = re.split(r"\s*", output[2].strip(), 1)
                goals.append(goal.read(output[0].strip(), decl[1], decl[0]))
            else:
                decl = re.split(r"\s*", output[0].strip(), 1)
                # this case makes sense, where the output is explicitly different to all inputs
                goals.append(goal.read(decl[1], decl[1], decl[0]))
    else:
        logging.error("No global outputs specified -- nothing to generate.")

    pg = rule_group()
    pg.rules += kernels
    pg.rules += code_blocks

    try:
        prefix = config["codegen options"]["prefix"]
    except KeyError:
        prefix = "__"

    language = config["codegen options"]["language"]
    vector_var = None
    try:
        vector_var = config["codegen options"]["vector loop"]
        if vector_var == "None":
            vector_var = None
        else:
            vector_var = symbolic_constant(vector_var)
    except KeyError:
        pass

    if language == "C" or language == "C99":
        generator = c99_generator
    elif language == "C++":
        generator = cpp_generator
    else:
        logging.error("Unrecognized language: %s -- select one of C, C99 or C++")

    if debug_output:
        cgen = generator(hfavroot, storage, None)
        cgen.debug_vector_var = vector_var
    else:
        cgen = generator(hfavroot, storage, vector_var)

    default_typedict = cgen.typedict.copy()
    try:
        for k, v in config["codegen options"]["types"].items():
            if k in cgen.typedict:
                logging.warning("%s already exists in dictionary -- overriding with %s.", k, v)
            m = re.match(r"([a-zA-Z]+)(\d+)?", v)
            if m is None:
                raise SyntaxError("Malformed type: %s -- expected <type><width>" % v)
            elif m.group(2) == None:
                if m.group(1) in default_typedict.keys():
                    cgen.typedict[k] = list(default_typedict[m.group(1)])
                else:
                    raise SyntaxError("Malformed type: %s -- width must be specified for all types not in %s" % (v, default_typedict.keys()))
            else:
                if m.group(1) not in ["int", "float"]:
                    raise SyntaxError("Malformed type: %s -- base type must be 'int' or 'float'" % m.group(1))
                cgen.typedict[k] = [m.group(1), int(m.group(2))]
    except KeyError:
        pass
    logging.debug("Using type dictionary: %s", cgen.typedict)

    loops = iteration_space.from_yaml(config)

    if args.output_location:
        if args.output_location == '-':
            of = sys.stdout
            logging.info("Generating code to stdout (overriden)")
        else:
            of = open(args.output_location, "w")
            output = args.output_location
            logging.info("Generating code to %s (overriden)", output)
    else:
        try:
            output = config["codegen options"]["output file"]
            of = open(output, "w")
            logging.info("Generating code into %s", output)
        except KeyError:
            of = sys.stdout
            logging.info("Generating code to stdout")

    try:
        header = config["codegen options"]["header"]
    except KeyError:
        header = None

    try:
        footer = config["codegen options"]["footer"]
    except KeyError:
        footer = None

    logging.info("Loaded input file")

    for iv in loops.loop_order:
        axioms.append(ivar_axiom(iv))

    logging.info("Chaining...")
    gr = dag_chain(pg, cgen.typedict, axioms).resolve(goals)
    logging.info("Chaining finished.")
    logging.info("IDAG has %s", gr.stats())
    logging.info("     Iteration space is over %s", [str(x) for x in gr.ivars()])

    rd = rap_dual.from_idag(gr)
    logging.info("Rap DUAL! %s ", rd.stats())
    order = rd.level_sort()
    for i, o in enumerate(order):
        logging.info("RD %d %s ", i, o.name())
    levels = rd.level_sort_levels()
    for i, l in enumerate(levels):
        logging.info("RD level %d %s ", i, [o.name() for o in l])
    rd.check_reductions()

    rap_loops = rd.topo_sort(lambda x: (len(x.rap_ivars()), x.rap_ivars()))
    for i, r in enumerate(rap_loops):
        logging.debug("%d %s %s", i, str(r), r.rap_ivars())

    if extra_output > 0:
        (root, ext) = os.path.splitext(os.path.basename(filename))
        dagfile = root + "rapdual.dot"
        logging.info("Writing out rapdual dag to %s", dagfile,)
        with file(dagfile, "w") as fi:
            print >> fi, rd.dot(v_fmt=lambda x: "%s-%s" % (x.name(), [str(i) for i in x.rap_ivars()]), e_fmt=lambda x: "")
        logging.info("Done writing out rapdual dag.")
    else:
        logging.info("Skipping writing rapdual dag.")

    if extra_output > 0:
        (root, ext) = os.path.splitext(os.path.basename(filename))
        dagfile = root + ".dot"
        logging.info("Writing out inference dag to %s", dagfile)
        with file(dagfile, "w") as fi:
            print >> fi, gr.dot()
        logging.info("Done writing out inference dag.")
    else:
        logging.info("Skipping writing inference dag.")

    fusion = not debug_output
    logging.info("Rap dual super node fusion.")
    indag = inest_dag(rd, loops)
    if extra_output > 0:
        (root, ext) = os.path.splitext(os.path.basename(filename))
        dagfile = root + "_inest.dot"
        logging.info("Writing out inest dag to %s", dagfile)
        with file(dagfile, "w") as fi:
            print >> fi, indag.dot(lambda v: str(v.inest), lambda v: "")
        logging.info("Done writing out inest dag.")
    else:
        logging.info("Skipping writing inest dag.")

    if not debug_output:
        logging.info("Fusing inest_dag")
        indag.topo_fuse()
        if extra_output > 0:
            (root, ext) = os.path.splitext(os.path.basename(filename))
            dagfile = root + "_inest_fused.dot"
            logging.info("Writing out fused dag to %s", dagfile)
            with file(dagfile, "w") as fi:
                print >> fi, indag.dot(lambda v: str(v.inest), lambda v: "")
            logging.info("Done writing out fused dag.")
        else:
            logging.info("Skipping writing fused dag.")
    else:
        logging.info("Not fusing inest_dag")

    rolling = True
    if rolling and fusion:
        logging.info("Preparing rolling generator")
        generator = rolling_generator
    else:
        logging.info("Preparing simple generator")
        generator = simple_generator

    ig = generator(indag, loops, cgen, prefix)
    logging.info("Generator initialized")
    lst = codegen.listing()
    cgen.header(lst, header)
    logging.info("Generating")
    ig.generate(lst)
    logging.info("Done generating")
    cgen.footer(lst, footer)
    if of != sys.stdout:
        logging.info("Writing code to %s", os.path.abspath(output))
    else:
        logging.info("Writing code to stdout")

    print >> of, lst.emit()

    if of != sys.stdout:
        of.close()

    logging.info("Finished generating code.")
    logging.info("Done; exiting.")

    sys.exit(0)

if __name__ == '__main__':
    hfav_run_yaml()
