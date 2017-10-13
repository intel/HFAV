# hfav/analyze.py; Use inference dag to analyize computation and guide code generation

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

import itertools as it
import operator as op
import logging
from copy import deepcopy
from operator import attrgetter
from operator import itemgetter
import sys
import re

from hfav.ispace import strided_interval, iteration_space
from hfav.iter_plot import iter_plot, iter_plot_start, iter_plot_finish
from hfav.infer import vectorize_symbolic_constant, vertex
from hfav.term import iteration_permutation, position
from hfav.inest import inest, inest_phases, inest_leaf, split_error
from . import infer
from . import codegen
from . import extra_output
from . import term
from . import dag

global logger
logger = logging.getLogger(__name__)

global path_colors
path_colors = ["#4D4D4D", "#5DA5DA", "#FAA43A", "#60BD68", "#F17CB0", "#B2912F", "#B276B2", "#DECF3F", "#F15854"]


# This is bad too. We shouldn't be using the key as the c name, probably
global var_key_noarray
var_key_noarray = infer.var_key_noarray


def aggregate_vars(v):
    """Extract all the unique variables in v"""
    res = {}
    for (n, z) in group_vars(v):
        pos = []
        for i in z:
            pos.append(i.position())
        res[n] = pos
    return res


def cut_key(it, d):
    if d >= len(it):
        return (), None
    return (it[:max(0, d)] + it[d + 1:]), it[d]


def stride_vec(delta, strides):
    diff = []
    sign = 0.0
    assert len(l) == len(r)
    assert len(r) == len(s)
    for (lv, rv, st) in it.izip(l, r, s):
        d = -(rv - lv) / float(st)
        if sign == 0.0 and d != 0.0:
            sign = -d / abs(d)
        diff.append(sign * d)
    if sign == -1:
        return (l, r, tuple(diff))
    else:
        return (r, l, tuple(diff))


class iter_edge(object):
    low_boundary = 0
    interior = 1
    high_boundary = 2


def iter_seq(perm, o):
    return tuple(reversed([o[perm[i]] for i in range(len(o))]))


def vector_orient(triple):
    (start, end, diff) = triple
    sign = 0.0
    nd = []
    for v, d in reversed(diff):
        if sign == 0.0 and d != 0.0:
            sign = -d / abs(d)
        nd.append((v, sign * d))
    if sign == -1:
        return (end, start, tuple(reversed(nd)))
    else:
        return triple


def vector_filter(triple):
    (start, end, diff) = triple
    nd = []
    for v, d in diff:
        if d != int(d):
            return False
    return True


def iter_vectors(dag_refs, strides):
    for li in range(len(dag_refs)):
        l = dag_refs[li]
        for ri in range(li + 1, len(dag_refs)):
            r = dag_refs[ri]
            v = vector_orient((r, l, l.difference(r, strides)))
            if vector_filter(v):
                yield v
    raise StopIteration()


def rolling_size(path, strides):
    # todo: replace this with something better
    # We assume that all members of the path have the same storage order
    dim_seen = dict((s[0], set()) for s in strides)
    for v in path.vertices.keys():
        for d, o in v.items:
            dim_seen[d].add(o)
    return dict((k, len(v)) for (k, v) in dim_seen.items())


class reuse_path(object):

    def __init__(self, path, loops, storage_order):
        self.path = path
        self.rolling_size = rolling_size(self.path, loops.strides())
        self.global_storage_order = storage_order

        # Construct map of global => local indices.
        # v.items() is in global_storage_order
        self.offsets = {}
        for d, iter_var in enumerate(self.global_storage_order):
            coords = []
            for v in self.path.vertices.keys():
                coords.append(int(v.items[d][1]))
                coords = sorted(set(coords))
                self.offsets[iter_var] = dict(zip(coords, range(len(coords))))

        # Determine the outer-most loop for which this variable has reuse.
        self.roll_var = None
        for d, iter_var in reversed(list(enumerate(loops.loop_order))):
            if self.rolling_size[iter_var] > 1 or self.rolling_size[iter_var] == 0:
                self.roll_var = iter_var
                break

        # Storage order for temporaries = loop order, but stripped of unnecessary indices.
        # (i.e. indices before the rolling one, or indices not used in the global storage)
        self.local_storage_order = []
        started_rolling = False
        for iter_var in reversed(loops.loop_order):
            if not started_rolling and iter_var == self.roll_var:
                started_rolling = True
            if started_rolling:
                if self.rolling_size[iter_var] != 0:
                    self.local_storage_order.append(iter_var)
        self.local_storage_order = list(reversed(self.local_storage_order))
        # print "%s + %s + %s => %s" % (self.rolling_size, self.roll_var, self.global_storage_order, self.local_storage_order)

    def outer_rolling(self):
        return len(self.local_storage_order) > 0 and self.roll_var != self.local_storage_order[0] and self.rolling_size[self.roll_var] > 1

    # TODO: This is an ugly function.
    def active_offsets(self, pos):
        if self.roll_var is not None:
            ao = []
            for iter_var in self.local_storage_order:
                found = False
                for x in pos.items:
                    if x[0] == iter_var:
                        ao.append(self.offsets[iter_var][x[1]])
                        found = True
                if not found:
                    ao.append(0)
            return ao
        else:
            return []


class reuse_dag(dag.dag):

    """Re-use/flow graph for a set of spatial references."""

    def __init__(self, positions, ispace):
        dag.dag.__init__(self)
        self.positions = positions
        self.ispace = ispace
        # TODO: Get Jason to do this in a smarter way.
        tmp_dag = dag.dag.from_edgelist([s, e] for (s, e, d) in list(iter_vectors(positions, ispace.strides())))
        for v in tmp_dag.vertices.values():
            self.add_vertex(v)
        for e in tmp_dag.edges.values():
            self.add_edge(e)
        for p in positions:
            try:
                self.add_vertex(self.vtype(p))
            except ValueError:
                pass

    def reuse_paths(self, loop_order):
        """Construct a list of separate reuse_paths for a flow graph, for a given loop order."""

        comps = self.longest_path_components()
        paths = [reuse_path(d, self.ispace, loop_order) for d in comps]
        return paths

    def zone_clip(self, zone):
        if len(self.vertices.keys()) > 0:
            perm, used = iteration_permutation((z[0] for z in zone), self.vertices.keys()[0].var_order())
        new_path = dag.dag()
        for v in self.vertices.values():
            new_path.add_vertex(v)
        for k, d in self.edges.items():
            s, e = k
            diff = e.difference(s)
            okay = True
            for i in range(len(diff)):
                ie = zone[i][1]
                oe = diff[perm[i]][1]
                if ie == iter_edge.low_boundary and oe < 0:
                    okay = False
                    break
                elif ie == iter_edge.high_boundary and oe > 0:
                    okay = False
                    break
            if okay:
                new_path.add_edge(d)
        return new_path


def var_dim_sizes(positions, strides):
    # todo: replace this with something better
    dim_seen = dict((s[0], set()) for s in strides)
    for p in positions:
        for d, o in p.items:
            dim_seen[d].add(o)
    return dict((k, len(v)) for (k, v) in dim_seen.items())


class variable_base(object):

    def __init__(self, prefix, name, var, loops, typedict):

        self.name = prefix + name
        self.ispace = loops
        self.family = []
        self.type = var[0][0]
        self.storage_order = var[0][1].position().var_order()

        self.positions = set([])
        for (typ, term) in var:
            assert typedict[typ] == typedict[self.type]
            fam_pos = term.position()
            assert self.storage_order == fam_pos.var_order()
            self.positions.add(fam_pos)
            self.family.append(term)
        self.positions = list(self.positions)
        self.live_positions = []

        # Global sizes (without reuse analysis).
        widths = var_dim_sizes(self.positions, loops.strides())
        self.sizes = []
        for d, iter_var in enumerate(self.storage_order):
            iterations = "%s-%s" % (loops.loop_dict[iter_var].end, loops.loop_dict[iter_var].start)
            width = widths[iter_var]
            if width == 1:
                size = "%s" % iterations  # (R == W-1) => (W-R)*N+R == N+R
            else:
                size = "%s+%d" % (iterations, width - 1)  # (R == W-1) => (W-R)*N+R == N+R
            self.sizes.append(size)
        self.sizes = list(reversed(self.sizes))

        # Mapping from iteration to index.
        self.offsets = {}
        for d, iter_var in enumerate(self.storage_order):
            coords = []
            for p in self.positions:
                coords.append(int(p.items[d][1]))
                coords = sorted(set(coords))
                self.offsets[iter_var] = dict(zip(coords, range(len(coords))))

        # Variable reuse analysis.
        # This is not only used for rolling -- it also tells us how to avoid redundant work!
        rdag = reuse_dag(self.positions, loops)
        self.rpaths = rdag.reuse_paths(self.storage_order)
        self.vpaths = {}
        for rp in self.rpaths:
            for pos in rp.path.vertices.keys():
                self.vpaths[pos] = rp

        self.zone_dags = {}
        for zone in [tuple(it.izip(loops.loop_order, x)) for x in it.product(*it.repeat(range(3), loops.dim()))]:
            self.zone_dags[zone] = rdag.zone_clip(zone)

        kl = len(self.zone_dags.keys()[0])
        for k in self.zone_dags.keys():
            assert kl == len(k)

        self.rolling_sizes = {}

    def store_ident(self):
        return self.name

    def access_ident(self):
        return self.name

    def outrefp(self):
        return "outref" in self.name

    def declarations(self, lst, cgen):
        if self.outrefp():
            return
        vsizes = list(self.sizes)
        if cgen.vector_var is not None:
            if self.storage_order == []:
                vsizes = ["VLEN"]
            elif cgen.vector_var == self.storage_order[0]:
                vsizes[-1] = str(vsizes[-1]) + "+VLEN-1"
            else:
                vsizes.append("VLEN")
        cgen.statement(lst, cgen.array_declaration(self.type, self.store_ident(), vsizes))

    def frees(self, lst, cgen):
        if self.outrefp():
            return
        vsizes = list(self.sizes)
        if cgen.vector_var is not None:
            if self.storage_order == []:
                vsizes = ["VLEN"]
            elif cgen.vector_var == self.storage_order[0]:
                vsizes[-1] = str(vsizes[-1]) + "+VLEN-1"
            else:
                vsizes.append("VLEN")
        cgen.statement(lst, cgen.array_free(self.type, self.store_ident(), vsizes))

    def active_offsets(self, pos):
        ao = []
        for iter_var in self.storage_order:
            found = False
            for x in pos.items:
                if x[0] == iter_var:
                    ao.append(self.offsets[iter_var][x[1]])
                    found = True
            if not found:
                ao.append(0)
        return ao

    def read_ref(self, var, cgen):
        pos = var.position()
        offsets = self.active_offsets(pos)
        for d, o in enumerate(offsets):
            offsets[d] = self.ispace.map_offset(self.storage_order[d], o)
        if cgen.vector_var is not None:
            if self.storage_order == []:
                offsets = ["__hfav_vlane"] if cgen.vectorize else ["0"]
            elif cgen.vector_var == self.storage_order[0]:
                offsets[0] = str(offsets[0]) + "+__hfav_vlane" if cgen.vectorize else str(offsets[0])
            else:
                offsets.insert(0, "__hfav_vlane" if cgen.vectorize else "0")
        return cgen.read_aref(self.access_ident(), reversed(offsets))

    def write_ref(self, var, cgen, byref=False):
        pos = var.position()
        offsets = self.active_offsets(pos)
        for d, o in enumerate(offsets):
            offsets[d] = self.ispace.map_offset(self.storage_order[d], o)
        if cgen.vector_var is not None:
            if self.storage_order == []:
                offsets = ["__hfav_vlane"] if cgen.vectorize else ["0"]
            elif cgen.vector_var == self.storage_order[0]:
                offsets[0] = str(offsets[0]) + "+__hfav_vlane" if cgen.vectorize else str(offsets[0])
            else:
                offsets.insert(0, "__hfav_vlane" if cgen.vectorize else "0")
        if byref:
            ident = cgen.byref(self.access_ident())
        else:
            ident = self.access_ident()
        return cgen.write_aref(ident, reversed(offsets))

    def rotations(self, lst, cgen, loop_var):
        return

    def is_active(self, zone, var):
        return True


class simple_variable(variable_base):

    def __init__(self, prefix, name, var, loops, typedict):
        super(simple_variable, self).__init__(prefix, name, var, loops, typedict)
        # print "created %s as a simple variable" % (self.name)


class simple_input_variable(variable_base):

    def __init__(self, prefix, name, var, loops, typedict):
        super(simple_input_variable, self).__init__(prefix, name, var, loops, typedict)

    def declarations(self, lst, cgen):
        pass

    def frees(self, lst, cgen):
        pass

    def read_ref(self, var, cgen):
        if cgen.vectorize:
            offsets = [vectorize_symbolic_constant(v, cgen.vector_var) for v in var.position_tuple()]
        else:
            offsets = var.position_tuple()
        return cgen.read_aref(self.access_ident(), reversed(offsets))

    def write_ref(self, var, cgen, byref=False):
        raise RuntimeError("Can't write to input_variable %s!", self.name)


class simple_output_variable(variable_base):

    def __init__(self, prefix, name, var, loops, typedict):
        super(simple_output_variable, self).__init__(prefix, name, var, loops, typedict)

    def declarations(self, lst, cgen):
        pass

    def frees(self, lst, cgen):
        pass

    def read_ref(self, var, cgen):
        if cgen.vectorize:
            offsets = [vectorize_symbolic_constant(v, cgen.vector_var) for v in var.position_tuple()]
        else:
            offsets = var.position_tuple()
        return cgen.read_aref(self.access_ident(), reversed(offsets))

    def write_ref(self, var, cgen, byref=False):
        if cgen.vectorize:
            offsets = [vectorize_symbolic_constant(v, cgen.vector_var) for v in var.position_tuple()]
        else:
            offsets = var.position_tuple()
        if byref:
            ident = cgen.byref(self.store_ident())
        else:
            ident = self.store_ident()
        return cgen.write_aref(ident, reversed(offsets))


def compute_rolling_sizes(var, loops):

    widths = var_dim_sizes(var.positions, loops.strides())
    rolling_sizes = {}
    for rp in var.rpaths:
        rolling_sizes[rp] = []
        for d, iter_var in enumerate(rp.local_storage_order):
            if rp.roll_var == iter_var:
                size = rp.rolling_size[iter_var]
            else:
                iterations = "%s-%s" % (loops.loop_dict[iter_var].end, loops.loop_dict[iter_var].start)
                width = widths[iter_var]
                if width == 1:
                    size = "%s" % iterations
                else:
                    size = "%s+%d" % (iterations, width - 1)
            rolling_sizes[rp].append(size)
        rolling_sizes[rp] = list(reversed(rolling_sizes[rp]))
    return rolling_sizes


class rolling_variable(variable_base):

    def __init__(self, prefix, name, var, loops, typedict):
        super(rolling_variable, self).__init__(prefix, name, var, loops, typedict)
        # print "created %s as a rolling variable" % (self.name)
        self.rolling_sizes = compute_rolling_sizes(self, loops)

        if extra_output > 1:
            if len(rdag.vertices.keys()) >= 1:
                reuse_dot = "%s_reuse.dot" % (name,)
                logger.info("Dumping out %s reuse graph to %s", name, reuse_dot)
                with open(reuse_dot, "w") as fp:
                    print >> fp, rdag.dot()
                if loops.dim() < 3:
                    try:
                        ax = iter_plot_start()
                        iter_plot(ax, rdag,
                                  vstyle={"ec": "none", "fc": path_colors[0], "zorder": 0},
                                  estyle={"ls": "dotted", "fc": path_colors[0], "ec": path_colors[0], "zorder": 1})
                        for pc, path_flow in it.izip(it.chain(path_colors[1:], it.repeat("#FF0000")), self.rpaths):
                            path = path_flow.path
                            if not path.empty():
                                iter_plot(ax, path,
                                          vstyle={"ec": "none", "fc": pc, "zorder": 2},
                                          estyle={"ls": "solid", "fc": pc, "ec": pc, "zorder": 3})
                        iter_plot_finish(name + "_reuse.png")
                        logger.info("Generating plot of %s as %s", name, name + "_reuse.png")
                    except:
                        logger.warning("Skipping plot of %s as %s", name, name + "_reuse.png")
            else:
                logger.info("Skipping %s reuse dot", name)

    def path_ident(self, rp):
        return "%s_p%d" % (self.name, self.rpaths.index(rp)) if len(self.rpaths) > 1 else self.name

    def store_ident(self, rp):
        return self.path_ident(rp) + ("_store" if rp.outer_rolling() else "")

    def access_ident(self, rp):
        return self.path_ident(rp) + ("_ptr" if rp.outer_rolling() else "")  # TODO: Decide whether we really want to append "ptr" or whether just "store" is enough.

    def declarations(self, lst, cgen):
        # assert(not self.outrefp()) # TODO: This should go back in
        if self.outrefp():
            return
        for (i, rp) in enumerate(self.rpaths):
            vsizes = list(self.rolling_sizes[rp])
            if cgen.vector_var is not None:
                if rp.local_storage_order == []:
                    vsizes = ["VLEN"]
                elif cgen.vector_var == rp.local_storage_order[0]:
                    vsizes[-1] = str(vsizes[-1]) + ("" if rp.outer_rolling() else "+VLEN-1")
                else:
                    vsizes.append("VLEN")
            cgen.statement(lst, cgen.array_declaration(self.type, self.store_ident(rp), vsizes))
            if rp.outer_rolling():
                cgen.statement(lst, cgen.array_ptr_declaration(self.type, self.access_ident(rp), self.store_ident(rp), vsizes))

    def frees(self, lst, cgen):
        if self.outrefp():
            return
        for (i, rp) in enumerate(self.rpaths):
            vsizes = list(self.rolling_sizes[rp])
            if cgen.vector_var is not None:
                if rp.local_storage_order == []:
                    vsizes = ["VLEN"]
                elif cgen.vector_var == rp.local_storage_order[0]:
                    vsizes[-1] = str(vsizes[-1]) + ("" if rp.outer_rolling() else "+VLEN-1")
                else:
                    vsizes.append("VLEN")
            cgen.statement(lst, cgen.array_free(self.type, self.store_ident(rp), vsizes))

    def read_ref(self, var, cgen):
        pos = var.position()
        rp = self.vpaths[pos]
        offsets = rp.active_offsets(pos)
        for d, o in enumerate(offsets):
            offsets[d] = self.ispace.map_offset(rp.local_storage_order[d], o, rp.roll_var)
        if cgen.vector_var is not None:
            if rp.local_storage_order == []:
                offsets = ["__hfav_vlane"] if cgen.vectorize else ["0"]
            elif cgen.vector_var == rp.local_storage_order[0]:
                offsets[0] = str(offsets[0]) + "+__hfav_vlane" if cgen.vectorize else str(offsets[0])
            else:
                offsets.insert(0, "__hfav_vlane" if cgen.vectorize else "0")
        return cgen.read_aref(self.access_ident(rp), reversed(offsets))

    def write_ref(self, var, cgen, byref=False):
        pos = var.position()
        rp = self.vpaths[pos]
        offsets = rp.active_offsets(pos)
        for d, o in enumerate(offsets):
            offsets[d] = self.ispace.map_offset(rp.local_storage_order[d], o, rp.roll_var)
        if cgen.vector_var is not None:
            if rp.local_storage_order == []:
                offsets = ["__hfav_vlane"] if cgen.vectorize else ["0"]
            elif cgen.vector_var == rp.local_storage_order[0]:
                offsets[0] = str(offsets[0]) + "+__hfav_vlane" if cgen.vectorize else str(offsets[0])
            else:
                offsets.insert(0, "__hfav_vlane" if cgen.vectorize else "0")
        if byref:
            ident = cgen.byref(self.access_ident(rp))
        else:
            ident = self.access_ident(rp)
        return cgen.write_aref(ident, reversed(offsets))

    def rotations(self, lst, cgen, loop_var):
        # assert(not self.outrefp()) # TODO: This should go back in
        if self.outrefp():
            return
        for rp in self.rpaths:
            if rp.roll_var is None:
                pass
            else:
                if rp.roll_var == loop_var and rp.rolling_size[rp.roll_var] > 0:

                    if rp.outer_rolling():
                        cgen.statement(lst, cgen.rotate_ptr(self.type, self.access_ident(rp), rp.rolling_size[loop_var], loop_var))
                        return

                    # live_offsets is for the whole variable family, not this specific reuse_path.
                    # TODO: Make live_offsets a per-path thing?
                    live_offsets = []
                    for pos in filter(lambda p: p in rp.path.vertices.keys(), self.live_positions):
                        for x in pos.items:
                            if x[0] == loop_var:
                                live_offsets.append(rp.offsets[loop_var][x[1]])
                    if min(live_offsets) != max(live_offsets):
                        cgen.statement(lst, cgen.rotate(self.type, self.access_ident(rp), min(live_offsets), max(live_offsets), loop_var))

    def is_active(self, zone, var):

        # Adjust zone for "slices" of lower dimensionality than the loop.
        # Roots of the reuse DAG clipped for this zone haven't been seen before.
        position = var.position()
        zone = tuple(sorted(zone, key=lambda z: self.ispace.loop_order.index(z[0])))
        return position in self.zone_dags[zone].roots()


class rolling_terminal_variable(rolling_variable):

    def __init__(self, prefix, name, var, loops, typedict):
        super(rolling_terminal_variable, self).__init__(prefix, name, var, loops, typedict)

    def declarations(self, lst, cgen):
        pass

    def frees(self, lst, cgen):
        pass

    def rotations(self, lst, cgen, loop_var):
        pass

    def read_ref(self, var, cgen):
        if cgen.vectorize:
            offsets = [vectorize_symbolic_constant(v, cgen.vector_var) for v in var.position_tuple()]
        else:
            offsets = var.position_tuple()
        return cgen.read_aref(self.name, reversed(offsets))


class rolling_input_variable(rolling_terminal_variable):

    def __init__(self, prefix, name, var, loops, typedict):
        super(rolling_input_variable, self).__init__(prefix, name, var, loops, typedict)

    def write_ref(self, var, cgen, byref=False):
        raise RuntimeError("Can't write to input_variable %s!", self.name)


class rolling_output_variable(rolling_terminal_variable):

    def __init__(self, prefix, name, var, loops, typedict):
        super(rolling_output_variable, self).__init__(prefix, name, var, loops, typedict)

    def write_ref(self, var, cgen, byref=False):
        if cgen.vectorize:
            offsets = [vectorize_symbolic_constant(v, cgen.vector_var) for v in var.position_tuple()]
        else:
            offsets = var.position_tuple()
        if byref:
            ident = cgen.byref(self.name)
        else:
            ident = self.name
        return cgen.write_aref(ident, reversed(offsets))


class iter_var(variable_base):

    def __init__(self, prefix, name, var, loops, typedict):
        super(iter_var, self).__init__(prefix, name, var, loops, typedict)

    def declarations(self, lst, cgen):
        pass

    def frees(self, lst, cgen):
        pass

    def read_ref(self, var, cgen):
        return cgen.read_ref(var)

    def write_ref(self, var, cgen, byref=False):
        raise RuntimeError("Can't write to iter_vars!")


def make_variable(varset, prefix, n, membs, loops, inputs, outputs, typedict):

    if loops.is_iter(n):
        return iter_var(prefix, n, membs, loops, typedict)

    if n in inputs:
        return varset.input_variable_type("", inputs[n], membs, loops, typedict)
    elif n in outputs:
        return varset.output_variable_type("", outputs[n], membs, loops, typedict)
    else:
        return varset.variable_type(prefix, n, membs, loops, typedict)


class rap_dual(dag.dag):

    """A dataflow DAG of the rule applications (RAPS) as vertices and variables as edges"""

    def __init__(self):
        super(rap_dual, self).__init__()

    @classmethod
    def from_idag(cls, idag):
        res = cls()
        unused = None
        for rap in idag.raps():
            res.add_vertex(rap)
            for ante in rap.antecedent:
                srcvert = idag.vertices[ante]
                incoming = idag.incoming_edges(srcvert.key())
                uraps = set(idag.edges[i].rap for i in incoming)
                for ar in uraps:
                    if (ar, rap) in res.edges:
                        if not srcvert in res.edges[(ar, rap)].terms:
                            res.edges[(ar, rap)].terms.append(srcvert)
                    else:
                        e = res.etype(ar, rap)
                        e.terms = [srcvert]
                        res.add_edge_spec(e)
            # we don't add most consequents, since they have been added by symmetry above
            for cons in rap.consequent:
                srcvert = idag.vertices[cons]
                outgoing = idag.outgoing_edges(srcvert.key())
                # We add consequents that are not stores so that we have space for them in the analysis
                # ultimately, they are unused. We should probably warn when all of these are unused.
                if len(list(outgoing)) == 0 and not rap.endpoint():
                    if unused is None:
                        unused = res.add_vertex(infer.unused_app())
                    if (rap, unused) in res.edges:
                        assert not srcvert in res.edges[(rap, unused)].terms
                        unused.types[cons] = rap.get_type(cons)
                        res.edges[(rap, unused)].terms.append(srcvert)
                    else:
                        e = res.etype(rap, unused)
                        e.terms = [srcvert]
                        unused.types[cons] = rap.get_type(cons)
                        res.add_edge_spec(e)
        res.check()
        return res

    # TODO: Sorry Jason.
    def check_reductions(self):
        """Tie each reduction_initializer and reduction_finalizer to the rap that does the reduction"""
        for (s, e) in self.edges.keys():
            sv = self.vertices[s]
            ev = self.vertices[e]
            if isinstance(sv, infer.reduction_initializer_app):
                sv.reduction_app = ev
            if isinstance(ev, infer.reduction_finalizer_app):
                ev.reduction_app = sv
        for rap in self.vertices.values():
            if (isinstance(rap, infer.reduction_initializer_app) or isinstance(rap, infer.reduction_finalizer_app)) and not hasattr(rap, 'reduction_app'):
                logging.error("Broken reduction chain; dangling rap: %s\n", rap)

    def load_vars(self):
        return set([var_key_noarray(rap.antecedent[0]).replace("_inref_", "") for rap in self.roots() if len(rap.antecedent) > 0])

    def store_vars(self):
        return set([var_key_noarray(rap.consequent[0]).replace("_outref_", "") for rap in self.leaves() if len(rap.consequent) > 0])

    def input_vars(self):
        """Find all global inputs that are not overwritten by output rules"""

        inputs = {}
        for rap in self.roots():
            if isinstance(rap, infer.load_app):
                ante = var_key_noarray(rap.antecedent[0]).replace("_inref_", "")
                if ante in self.load_vars() - self.store_vars():
                    cons = var_key_noarray(rap.consequent[0])
                    inputs[cons] = ante
        return inputs

    def output_vars(self):
        """Find all global outputs that do not require temporary storage"""

        outputs = {}
        for rap in self.leaves():
            if isinstance(rap, infer.store_app):
                cons = var_key_noarray(rap.consequent[0]).replace("_outref_", "")
                if cons in self.store_vars() - self.load_vars():
                    ante = var_key_noarray(rap.antecedent[0])
                    outputs[ante] = cons
        return outputs

    def inout_chains(self):
        """Use the inout values from raps to construct the 'chains' of connected inout terms"""

        io_chains = {}
        for rap in self.level_sort():
            if rap.terminal():
                continue
            for p, (s, e) in rap.inouts.items():
                if s.arg in io_chains:
                    assert not e.arg in io_chains
                    ch = io_chains[s.arg]
                    del io_chains[s.arg]
                else:
                    ch = [(s.arg, s.type, rap, p)]
                ch.append((e.arg, e.type, rap, p))
                io_chains[e.arg] = ch

        # Collapse chains that are related spatially.
        collapsed_chains = {}
        for (n, chk) in it.groupby(sorted(set(io_chains.keys()), key=lambda x: var_key_noarray(x)), key=lambda x: var_key_noarray(x)):
            ch = []
            for k in chk:
                ch += io_chains[k]
            collapsed_chains[n] = ch
        return collapsed_chains.values()


# TODO: Turn this into a container of variables.
class simple_chain_variable(variable_base):

    """Container for all variables in a chain.
    Re-use analysis must be performed globally and locally."""

    def __init__(self, prefix, chain, loops, typedict):
        representative_term = chain[-1][1]
        super(simple_chain_variable, self).__init__(prefix, var_key_noarray(representative_term), chain, loops, typedict)
        self.prefix = prefix
        self.representative_term = representative_term

    def term_vertex(self):
        return infer.vertex(self.representative_term, self.type)

    def is_active(self, zone, var):
        if "_reduction" in var_key_noarray(var):
            return True
        else:
            super(simple_chain_variable, self).is_active(zone, var)


# TODO: Turn this into a container of rolling variables.
class rolling_chain_variable(rolling_variable):

    """Container for all rolling variables in a chain.
    Re-use analysis must be performed globally and locally."""

    def __init__(self, prefix, chain, loops, typedict):
        representative_term = chain[-1][1]
        super(rolling_chain_variable, self).__init__(prefix, var_key_noarray(representative_term), chain, loops, typedict)
        self.prefix = prefix
        self.representative_term = representative_term

    def term_vertex(self):
        return infer.vertex(self.representative_term, self.type)

    def is_active(self, zone, var):
        if "_reduction" in var_key_noarray(var):
            position = var.position()
            ivars = position.named.keys()
            zone = tuple(sorted([z if (z[0] in ivars) else tuple([z[0], 0]) for z in zone], key=lambda k: self.ispace.loop_order.index(k[0])))
            return position in self.zone_dags[zone].roots()
        else:
            return super(rolling_chain_variable, self).is_active(self, zone, var)


class variable_set_base(object):

    """Container for all the variables used to generate code.
    Generates read/write refs, declarations. Base class"""

    def __init__(self, loops, prefix):
        self.prefix = prefix
        self.loops = loops
        self.variable_families = {}

    def process_chains(self, rap_dual, typedict):
        for ch in rap_dual.inout_chains():
            cv = self.chain_variable_type(self.prefix, [(link[1], link[0]) for link in ch], self.loops, typedict)
            term_vertex = cv.term_vertex()
            for v in cv.family:
                self.link_chain_variable(v, cv)
            for ((arg0, type0, rap0, p0), (arg1, type1, rap1, p1)) in it.izip(ch, ch[1:]):
                for e in rap_dual.incoming_edges(rap0):
                    rap_dual.edges[e].terms = [term_vertex if v.term == arg0 else v for v in rap_dual.edges[e].terms]
                for e in rap_dual.outgoing_edges(rap1):
                    rap_dual.edges[e].terms = [term_vertex if v.term == arg1 else v for v in rap_dual.edges[e].terms]

    def add_variable(self, n, members, inputs, outputs, typedict):
        assert n not in self.variable_families
        self.variable_families[n] = make_variable(self, self.prefix, n, members, self.loops, inputs, outputs, typedict)

    def link_chain_variable(self, v, cv):
        self.variable_families[var_key_noarray(v)] = cv

    def add_iter_variable(self, n, members, inputs, outputs, typedict):
        if n in self.variable_families:
            assert isinstance(self.variable_families[n], iter_var)
            return
        self.variable_families[n] = make_variable(self, self.prefix, n, members, self.loops, inputs, outputs, typedict)

    def make_live(self, v):
        if var_key_noarray(v) in self.variable_families:
            self.variable(v).live_positions.append(v.position())  # TODO: Should this be a set?

    def variable(self, var):
        vk = var_key_noarray(var)
        return self.variable_families[vk]

    def is_active(self, zone, var):
        return self.variable(var).is_active(zone, var)

    def is_input_variable(self, var):
        vf = self.variable(var)
        return any([isinstance(vf, type) for type in [simple_input_variable, rolling_input_variable, enclosing_input_variable]])

    def is_output_variable(self, var):
        vf = self.variable(var)
        return any([isinstance(vf, type) for type in [simple_output_variable, rolling_output_variable, enclosing_output_variable]])

    def is_rolling_variable(self, var):
        return isinstance(self.variable(var), rolling_variable)

    def is_chain_variable(self, var):
        return any([isinstance(vf, type) for type in [simple_chain_variable, rolling_chain_variable, enclosing_chain_variable]])

    def read_ref(self, v, cgen):
        return self.variable(v).read_ref(v, cgen)

    def write_ref(self, v, cgen, byref=False):
        return self.variable(v).write_ref(v, cgen, byref)

    def declarations(self, lst, cgen):
        for v in sorted(set(self.variable_families.values()), key=lambda x: x.name):
            v.declarations(lst, cgen)

    def frees(self, lst, cgen):
        for v in sorted(set(self.variable_families.values()), key=lambda x: x.name):
            v.frees(lst, cgen)

    def rotations(self, lst, cgen, loop_var):
        for v in sorted(set(self.variable_families.values()), key=lambda x: x.name):
            v.rotations(lst, cgen, loop_var)


class inner_variable_set_base(variable_set_base):

    def __init__(self, enclosing, rap_dual, loops, global_inputs, global_outputs, prefix, cg):
        super(inner_variable_set_base, self).__init__(loops, prefix)

        self.enclosing = enclosing

        inputs = rap_dual.input_vars()
        inputs.update(global_inputs)
        outputs = rap_dual.output_vars()
        outputs.update(global_outputs)

        self.process_chains(rap_dual, cg.typedict)

        logger.debug("processing edges")
        vars = []
        for (s, e), ev in rap_dual.edges.items():
            for v in ev.terms:
                vars.append((v.type, v.term))

        logger.debug("done processing edges")
        logger.debug("making variables")
        # of course, I don't have to sort here: I could just have variable_families be update-able. Oh well.
        for (n, membs) in it.groupby(sorted(set(vars), key=lambda x: var_key_noarray(x[1])), key=lambda x: var_key_noarray(x[1])):
            if n not in self.enclosing.variable_families and n not in self.variable_families:
                self.add_variable(n, list(membs), inputs, outputs, cg.typedict)

        # TODO: Why is this necessary?
        for iv in loops.loop_order:
            self.add_iter_variable(str(iv), [("int", iv)], inputs, outputs, cg.typedict)

        # An empty variable set is okay if all of the variables are enclosing -- otherwise something went wrong.
        # TODO: Seek Jason's blessing.
        if not self.variable_families and not all(n in self.enclosing.variable_families for n in [var_key_noarray(x[1]) for x in vars]):
            raise RuntimeError("Made an empty variable family!")

        logger.debug("done variables")

    def link_chain_variable(self, v, cv):
        if not var_key_noarray(v) in self.enclosing.variable_families:
            self.variable_families[var_key_noarray(v)] = cv

    def variable(self, var):
        try:
            return super(inner_variable_set_base, self).variable(var)
        except KeyError:
            return self.enclosing.variable(var)


class enclosing_variable(variable_base):

    def __init__(self, prefix, name, var, loops, typedict):
        super(enclosing_variable, self).__init__(prefix, name, var, loops, typedict)
        self.rolling_sizes = compute_rolling_sizes(self, loops)

    def is_active(self, zone, var):

        # Roots of the reuse DAG adjusted for this zone haven't been seen before.
        # TODO: Pythonify
        position = var.position()
        zone_map = {}
        for (itervar, z) in zone:
            zone_map.update({itervar:z})
        for itervar in self.ispace.loop_order:
            if not itervar in zone_map:
                zone_map.update({itervar:0})
        zone = tuple(sorted([tuple([itervar, z]) for (itervar, z) in zone_map.items()], key=lambda z: self.ispace.loop_order.index(z[0])))
        return position in self.zone_dags[zone].roots()


class enclosing_input_variable(enclosing_variable):

    def __init__(self, prefix, name, var, loops, typedict):
        super(enclosing_input_variable, self).__init__(prefix, name, var, loops, typedict)

    def declarations(self, lst, cgen):
        pass

    def frees(self, lst, cgen):
        pass

    def read_ref(self, var, cgen):
        if cgen.vectorize:
            offsets = [vectorize_symbolic_constant(v, cgen.vector_var) for v in var.position_tuple()]
        else:
            offsets = var.position_tuple()
        return cgen.read_aref(self.access_ident(), reversed(offsets))

    def write_ref(self, var, cgen, byref=False):
        raise RuntimeError("Can't write to input_variable %s!", self.name)


class enclosing_output_variable(enclosing_variable):

    def __init__(self, prefix, name, var, loops, typedict):
        super(enclosing_output_variable, self).__init__(prefix, name, var, loops, typedict)

    def declarations(self, lst, cgen):
        pass

    def frees(self, lst, cgen):
        pass

    def read_ref(self, var, cgen):
        if cgen.vectorize:
            offsets = [vectorize_symbolic_constant(v, cgen.vector_var) for v in var.position_tuple()]
        else:
            offsets = var.position_tuple()
        return cgen.read_aref(self.access_ident(), reversed(offsets))

    def write_ref(self, var, cgen, byref=False):
        if cgen.vectorize:
            offsets = [vectorize_symbolic_constant(v, cgen.vector_var) for v in var.position_tuple()]
        else:
            offsets = var.position_tuple()
        if byref:
            ident = cgen.byref(self.store_ident())
        else:
            ident = self.store_ident()
        return cgen.write_aref(ident, reversed(offsets))


# TODO: Turn this into a container of enclosing variables.
class enclosing_chain_variable(enclosing_variable):

    """Container for all enclosing variables in a chain."""

    def __init__(self, prefix, chain, loops, typedict):
        representative_term = chain[-1][1]
        super(enclosing_chain_variable, self).__init__(prefix, var_key_noarray(representative_term), chain, loops, typedict)
        self.prefix = prefix
        self.representative_term = representative_term

    def term_vertex(self):
        return infer.vertex(self.representative_term, self.type)

    def is_active(self, zone, var):
        # For reduction variables, re-use analysis should ignore the reduction iterators.
        if "_reduction" in var_key_noarray(var):
            position = var.position()
            ivars = position.named.keys()
            zone = tuple(sorted([z if (z[0] in ivars) else tuple([z[0], 0]) for z in zone], key=lambda k: self.ispace.loop_order.index(k[0])))
            return position in self.zone_dags[zone].roots()
        else:
            return super(enclosing_chain_variable, self).is_active(zone, var)


class enclosing_variable_set_base(variable_set_base):

    """An variable set that handles the storage of 'global' quantities that move between inner sets"""

    def __init__(self, inest_dag, loops, prefix, cg):
        """This needs to identify global inputs/ouputs, plus variables that connect inests.  It is responsible for providing storage/variable types for them"""

        super(enclosing_variable_set_base, self).__init__(loops, prefix)

        # keep in mind that 'input_vars' and 'output_vars' only return those terminal variables that are not not overwritten
        global_roots = inest_dag.source_dag.input_vars()
        global_leaves = inest_dag.source_dag.output_vars()
        logging.debug("ENCLOSING; global roots: %s", global_roots)
        logging.debug("ENCLOSING; global leaves: %s",  global_leaves)

        self.process_chains(inest_dag.source_dag, cg.typedict)

        # keep track of changing vertex names
        vmap = dict((v, v) for v in inest_dag.vertices.keys())

        # Find all of the variables in the original rap_dual.
        # TODO: Sorry Jason -- I don't like that we do this twice, but I didn't want to do a complete refactor until we had discussed it.
        vars = []
        for (s, e), ev in inest_dag.source_dag.edges.items():
            for v in ev.terms:
                vars.append((v.type, v.term))

        # Here, we identify all edges in the inest_dag (which corresponds to perhaps multiple variables)
        enclosing_vars = set([])
        original_edges = list(inest_dag.edges.items())
        logging.debug("ENCLOSING: processing %d inest_dag edges", len(original_edges))
        for ((os, oe), ev) in original_edges:
            for ce in ev.child_edges():

                # Delete the edge from the rap dual
                inest_dag.source_dag.remove_edge(ce.key())

                # These are edges from the rap_dual
                for v in ce.terms:

                    # Each of these terms is a variable that must be in the enclosing scope
                    enclosing_vars.add(var_key_noarray(v.term))
                    if var_key_noarray(v.term) in global_roots or var_key_noarray(v.term) in global_leaves:
                        continue

                    # If these are not roots/leaves to enclosing, we need to modify their references in rap_dual to have output/input refs
                    assert ce.terminals[0] in inest_dag.vertices[vmap[os]].inest.child_rapset()
                    assert ce.terminals[1] in inest_dag.vertices[vmap[oe]].inest.child_rapset()

                    # Add new output to the start point in the source dag
                    storing = infer.enclosing_store_app(v.term, v.type)
                    store_vert = inest_dag.source_dag.add_vertex(storing)
                    store_edge = inest_dag.source_dag.etype(ce.terminals[0], store_vert.key())
                    store_edge.terms = [v]
                    inest_dag.source_dag.add_edge(store_edge)

                    # Add new output to the inest_dag.
                    vt = inest_dag.add_vertex(inest_dag.vtype(inest.make_nest(storing.outer_ivars(), storing.inner_ivars(), inest_dag.ispace, set([storing]))))
                    et = inest_dag.etype(vmap[os], vt.key(), [store_edge])
                    inest_dag.add_edge(et)

                    # Try to fuse the new inests together.  This is expected to work except for codeblocks
                    try:
                        fused = inest_dag.fuse(vmap[os], vt.key())
                        vmap[vt.key()] = fused
                        vmap[os] = fused

                        # It seems like the error is coming from a lack of globals!!
                        assert store_edge.key() in inest_dag.source_dag.edges
                        fused_raps = inest_dag.vertices[fused].inest.child_rapset()
                        assert store_edge.terminals[0] in fused_raps
                        assert store_edge.terminals[1] in fused_raps
                        subgraph = inest_dag.source_dag.subgraph(fused_raps)
                        assert store_edge.key() in subgraph.edges
                    except split_error as se:
                        if isinstance(ce.terminals[0], infer.codeblock_app):
                            pass
                        else:
                            raise split_error(se.iident)

                    # Add new input to the endpoint
                    loading = infer.enclosing_load_app(v.term, v.type)
                    inest_dag.source_dag.add_vertex(loading)
                    load_edge = inest_dag.source_dag.etype(loading, ce.terminals[1])
                    load_edge.terms = [v]
                    inest_dag.source_dag.add_edge(load_edge)

                    vt = inest_dag.add_vertex(inest_dag.vtype(inest.make_nest(loading.outer_ivars(), loading.inner_ivars(), inest_dag.ispace, set([loading]))))
                    et = inest_dag.etype(vt.key(), vmap[oe], [load_edge])
                    inest_dag.add_edge(et)

                    # Try to fuse the new inests together.  This is expected to work except for codeblocks
                    try:
                        fused = inest_dag.fuse(vt.key(), vmap[oe])
                        vmap[vt.key()] = fused
                        vmap[oe] = fused
                    except split_error as se:
                        if isinstance(ce.terminals[1], infer.codeblock_app):
                            pass
                        else:
                            raise split_error(se.iident)

        # Create enclosing variables for subset of variables that require it.
        for (n, membs) in it.groupby(sorted(set(vars), key=lambda x: var_key_noarray(x[1])), key=lambda x: var_key_noarray(x[1])):
            if n in enclosing_vars and n not in self.variable_families:
                self.add_variable(n, list(membs), global_roots, global_leaves, cg.typedict)


class simple_enclosing_variable_set(enclosing_variable_set_base):

    def __init__(self, inest_dag, loops, prefix, cg):
        self.variable_type = simple_variable
        self.input_variable_type = simple_input_variable
        self.output_variable_type = simple_output_variable
        self.chain_variable_type = simple_chain_variable
        super(simple_enclosing_variable_set, self).__init__(inest_dag, loops, prefix, cg)


class rolling_enclosing_variable_set(enclosing_variable_set_base):

    def __init__(self, inest_dag, loops, prefix, cg):
        self.variable_type = enclosing_variable
        self.input_variable_type = enclosing_input_variable
        self.output_variable_type = enclosing_output_variable
        self.chain_variable_type = enclosing_chain_variable
        super(rolling_enclosing_variable_set, self).__init__(inest_dag, loops, prefix, cg)


class simple_variable_set(inner_variable_set_base):

    def __init__(self, enclosing, rap_dual, loops, global_inputs, global_outputs, prefix, cg):
        self.variable_type = simple_variable
        self.input_variable_type = simple_input_variable
        self.output_variable_type = simple_output_variable
        self.chain_variable_type = simple_chain_variable
        super(simple_variable_set, self).__init__(enclosing, rap_dual, loops, global_inputs, global_outputs, prefix, cg)


class rolling_variable_set(inner_variable_set_base):

    def __init__(self, enclosing, rap_dual, loops, global_inputs, global_outputs, prefix, cg):
        self.variable_type = rolling_variable
        self.input_variable_type = rolling_input_variable
        self.output_variable_type = rolling_output_variable
        self.chain_variable_type = rolling_chain_variable
        super(rolling_variable_set, self).__init__(enclosing, rap_dual, loops, global_inputs, global_outputs, prefix, cg)

    def add_variable(self, n, members, inputs, outputs, typedict):
        assert n not in self.variable_families
        self.variable_families[n] = make_variable(self, self.prefix, n, members, self.loops, inputs, outputs, typedict)


class generator_base(object):

    def __init__(self, inest_dag, ispace, cg, prefix="__hfav"):
        self.cgen = cg
        self.inest_dag = inest_dag
        self.ispace = ispace

        self.ev = self.enclosing_variable_set_type(inest_dag, self.ispace, prefix, cg)
        logging.debug("EVARS!!! %s", self.ev.variable_families)

        self.vset = {}
        ov = inest_dag.topo_sort()
        for v, inest in ((v, inest_dag.vertices[v].inest) for v in ov):
            vset = self.variable_set_type(self.ev, inest_dag.source_dag.subgraph(inest.child_rapset()), inest.ispace(), {}, {}, prefix, cg)
            np = self.adjust_nest(v, inest, inest_dag, vset)
            inest_dag.move_vertex(v, np)
            if np:
                self.vset[np] = vset
        self.source_dag_order = self.inest_dag.source_dag.level_sort(lambda rap: rap.sub)

    def adjust_nest(self, vert, inest, inest_dag, vset):
        np = inest.rap_zone(vset)
        return np

    def generate(self, lst):
        self.ev.declarations(lst, self.cgen)
        for d, inest in enumerate([self.inest_dag.vertices[v].inest for v in self.inest_dag.topo_sort()]):
            self.cgen.begin_scope(lst)
            self.cgen.init_iters(lst, inest.ispace())
            self.vset[inest].declarations(lst, self.cgen)
            inest.generate(lst, self.cgen, self.source_dag_order, self.vset[inest])
            self.vset[inest].frees(lst, self.cgen)
            self.cgen.end_scope(lst)
        self.ev.frees(lst, self.cgen)


class rolling_generator(generator_base):

    def __init__(self, inest_dag, ispace, cg, prefix="__hfav"):
        self.variable_set_type = rolling_variable_set
        self.enclosing_variable_set_type = rolling_enclosing_variable_set
        super(rolling_generator, self).__init__(inest_dag, ispace, cg, prefix)


class simple_generator(generator_base):

    def __init__(self, inest_dag, ispace, cg, prefix="__hfav"):
        self.variable_set_type = simple_variable_set
        self.enclosing_variable_set_type = simple_enclosing_variable_set
        super(simple_generator, self).__init__(inest_dag, ispace, cg, prefix)

    def adjust_nest(self, vert, inest, inest_dag, vset):
        np = super(simple_generator, self).adjust_nest(vert, inest, inest_dag, vset)
        if np:
            np = np.debug_redundant_adjust()
        return np
