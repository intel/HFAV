# hfav/inest.py; iteration nest manipulation tools

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

from copy import deepcopy as cdc
from copy import copy
import itertools as it
import collections
from hfav.dag import dag, vertex, edge, super_edge
from hfav.ispace import strided_interval, iteration_space
from . import term, extra_output
from . import infer
import logging
import sys

logger = logging.getLogger(__name__)


class inest_leaf(object):

    def __init__(self, rapset):
        self.rapset = rapset
        self.internal = False

    def outer_only(self):
        return True

    def __copy__(self):
        res = inest_leaf(self.rapset)
        assert(repr(self) != repr(res))
        return res

    def __eq__(self, o):
        return isinstance(o, inest_leaf) and self.rapset == o.rapset

    @classmethod
    def make_nest(cls, raps):
        return cls(raps)

    def rap_zone(self, vset, zone={}):
        """Ensure that the rapset for this leaf contains only raps in the specified zone."""

        # unused_app handling
        if zone == {}:
            return self

        zone_tuple = tuple(zip(zone.keys(), zone.values()))
        new_rapset = set()
        removed = 0
        for rap in self.rapset:
            if rap.in_zone(vset, zone_tuple):
                new_rapset |= set([rap])
            else:
                removed += 1
        self.rapset = new_rapset

        # TODO: Put this somewhere else.
        # Use the steady-state behaviour to build a list of live variables.
        if all([x[1] == 1 for x in zone_tuple]):
            for rap in self.rapset:
                for var in rap.antecedent + rap.consequent:
                    vset.make_live(var)

        if not self.rapset:
            logger.debug("Killing leaf")
            return None
        else:
            return self

    def __str__(self):
        return "[%s]" % "\n".join(map(str, self.rapset))

    def child_rapset(self):
        return self.rapset

    def ispace(self):
        return iteration_space({}, [])

    def perfect(self):
        return True

    def debug_redundant_adjust(self):
        """We shouldn't have multiple spatial raps in a leaf alone, so return self."""
        return self

    def generate(self, lst, cgen, source_dag_order, vset):
        if cgen.vectorize:
            cgen.begin_vector_loop(lst)
        for rap in it.ifilter(lambda x: x in self.rapset, source_dag_order):
            statement = rap.emit(vset, cgen)
            if isinstance(rap, infer.codeblock_app):
                lst.append(statement)
            elif statement is not None:
                cgen.statement(lst, statement)
        if cgen.vectorize:
            cgen.end_vector_loop(lst)


class inest_phases(object):

    def __init__(self, *args):
        assert len(args) == 3

        self.phases = [copy(args[0])]
        for v in args[1:]:
            if v == self.phases[-1]:
                self.phases.append(copy(self.phases[-1]))
            else:
                self.phases.append(copy(v))

        # make sure we don't somehow end up with an 'empty middle' with something on both ends
        assert not (self.phases[1] is None and (self.phases[0] is not None and self.phases[2] is not None))

    def empty(self):
        return all(x is None for x in self.phases)

    def unique(self):
        return [(x, [z[0] for z in y]) for (x, y) in it.groupby(enumerate(self.phases), lambda x: x[1])]

    def unsplit(self):
        return self.unsplit_prologue() and self.unsplit_epilogue()

    def outer_only(self):
        return all(x.outer_only() for x in self.phases)

    def perfect(self):
        """i.e., this and all children are unsplit"""
        return self.unsplit() and self.phases[0].perfect()

    def unsplit_prologue(self):
        return self.phases[0] == self.phases[1]

    def unsplit_epilogue(self):
        return self.phases[1] == self.phases[2]

    def child_rapset(self):
        res = set()
        for p in self.phases:
            res.update(p.child_rapset())
        return res

    def ispace(self):
        phase_dict = None
        phase_order = None
        for p in self.phases:
            subspace = p.ispace()
            if not phase_dict:
                phase_dict = subspace.loop_dict
            else:
                assert phase_dict.keys() == subspace.loop_dict.keys()
            if not phase_order:
                phase_order = subspace.loop_order
            else:
                assert phase_order == subspace.loop_order
        return iteration_space(subspace.loop_dict, subspace.loop_order)

    def prologue_prime(self):
        return self.phases[0].child_rapset().difference(self.phases[1].child_rapset())

    def epilogue_prime(self):
        return self.phases[2].child_rapset().difference(self.phases[1].child_rapset())

    def __eq__(self, o):
        return all(l == r for (l, r) in it.izip_longest(self.phases, o.phases))

    def __str__(self):
        return ", ".join(("%s: %s" % ("".join(("PSE"[x] for x in phase)), n) for (n, phase) in self.unique()))

    def __copy__(self):
        res = inest_phases(*[[], [], []])
        res.phases = [copy(p) for p in self.phases]  # res.phases = copy(self.phases) is subtly different, suggesting maybe we wanted a deepcopy.  Oh well.
        assert(repr(self) != repr(res))
        return res

    def __len__(self):
        return self.phases  # == 3

    def __getitem__(self, key):
        return self.phases[key]

    def __iter__(self):
        return iter(self.phases)

    def __repr__(self):
        return ", ".join([repr(self[x]) for x in range(3)])


class inest(object):

    def __init__(self, iident, internal, interval, phases):
        self.iident = iident
        self.internal = internal
        self.interval = interval
        self.phases = inest_phases(*phases)

    def outer_only(self):
        return not self.internal and self.phases.outer_only()

    def perfect(self):
        return self.phases.perfect()

    def __copy__(self):
        res = inest(self.iident, self.internal, self.interval, [[], [], []])
        #self.internal = False
        res.phases = copy(self.phases)
        assert(repr(self) != repr(res))
        return res

    def __eq__(self, o):
        return isinstance(o, inest) and self.iident == o.iident and self.interval == o.interval and self.phases == o.phases and self.internal == o.internal

    def child_rapset(self):
        return self.phases.child_rapset()

    def ispace(self):
        loop_dict = {self.iident: self.interval}
        loop_order = [self.iident]
        subspace = self.phases.ispace()
        loop_dict.update(subspace.loop_dict)
        loop_order = subspace.loop_order + loop_order
        return iteration_space(loop_dict, loop_order)

    @classmethod
    def make_nest_helper(cls, ex_idents, in_idents, intervals, raps):
        if len(ex_idents) + len(in_idents) == 0:
            return inest_leaf(raps)
        if len(ex_idents) > 0:
            return cls(ex_idents[0], False, intervals[0], [cls.make_nest_helper(ex_idents[1:], in_idents, intervals[1:], raps) for x in range(3)])
        else:
            return cls(in_idents[0], True, intervals[0], [cls.make_nest_helper([], in_idents[1:], intervals[1:], raps) for x in range(3)])

    @classmethod
    def make_nest(cls, outer, inner, ispace, raps):
        loop_order = list(reversed(ispace.loop_order))
        inner_order = [l for l in loop_order if l in inner]
        outer_order = [l for l in loop_order if l in outer]
        intervals = [ispace.loop_dict[l] for l in outer_order + inner_order]
        logger.debug("Making nest for %s: outer: %s, inner: %s" , raps, outer_order, inner_order)
        return cls.make_nest_helper(outer_order, inner_order, intervals, raps)

    def rap_zone(self, vset, zone={}):
        phasep = []
        for (p, n) in enumerate(self.phases):
            zone.update({self.iident: p})
            phasep.append(n.rap_zone(vset, zone))
            del zone[self.iident]
        self.phases = inest_phases(*phasep)
        if self.phases.empty():
            logger.debug("Killing phase")
            return None
        else:
            return self

    def __str__(self):
        return "%s {%s}" % (self.iident, self.phases)

    def generate(self, lst, cgen, source_dag_order, vset):
        for n, p in self.phases.unique():
            if not self.internal:
                cgen.begin_loop(lst, self.iident, self.interval, p)
            n.generate(lst, cgen, source_dag_order, vset)
            if not self.internal:
                vset.rotations(lst, cgen, self.iident)
                cgen.end_loop(lst, self.iident, self.interval, p)
                if self.iident == cgen.vector_var and 1 in p:
                    cgen.begin_remainder_loop(lst, self.iident, self.interval, p)
                    n.generate(lst, cgen, source_dag_order, vset)
                    vset.rotations(lst, cgen, self.iident)
                    cgen.end_remainder_loop(lst, self.iident, self.interval, p)

    def __repr__(self):
        return "%s: {%s}" % (super(inest, self).__repr__(), repr(self.phases))

    def debug_redundant_adjust(self):
        if self.perfect() and self.outer_only():
            ispace = self.ispace()
            distances = [infer.canonical_distances(d, ispace.loop_order) for d in infer.raplist_distances(list(self.child_rapset()))]
            compare_dist0 = [tuple((d[ind] for ind in ispace.loop_order)) for r, d in distances[0]]
            # This is problematic, we don't have a way to organize these right now
            for dist in distances[1:]:
                compare_distn = [tuple((d[ind] for ind in ispace.loop_order)) for r, d in dist]
                if compare_dist0 != compare_distn:
                    return self

            outer_iv = set([])
            rap_ispace = ispace.copy()
            for l in ispace.loop_order:
                ld = [dist[l] for rap, dist in distances[0]]
                soff = ("+" + str(min(ld))) if min(ld) != 0 else ""
                eoff = ("+" + str(max(ld))) if max(ld) != 0 else ""
                outer_iv.add(l)
                rap_ispace.loop_dict[l] = strided_interval(ispace.loop_dict[l].start + soff, ispace.loop_dict[l].end + eoff, ispace.loop_dict[l].stride)
            return inest.make_nest(outer_iv, set([]), rap_ispace, set([x[0][0] for x in distances]))
        return self


class inest_vertex(vertex):

    def __init__(self, inest):
        self.inest = inest

    def __repr__(self):
        return repr(self.inest)

    def key(self):
        """This is potentially dangerous, since obviously 'perfectly nested' inests will have the same child raps"""
        return tuple(sorted(self.inest.child_rapset()))


class inest_edge(edge):

    def __init__(self, s, e, constituents):
        super(inest_edge, self).__init__(s, e)
        self.constituents = constituents

        self.splits = set()

    def split(self, ident):
        self.splits.add(ident)

    def is_split(self):
        return len(self.splits) > 0

    def child_edges(self):
        return self.constituents


class inest_super_edge(super_edge):

    def __init__(self, s, e, constituents):
        super(inest_super_edge, self).__init__(s, e, constituents)

    def split(self, ident):
        for ev in self.constituents:
            ev.split(ident)

    def is_split(self):
        ## check to make sure things are all split the same????
        return any(x.is_split() for x in self.constituents)

    def child_edges(self):
        return it.chain(*(x.child_edges() for x in self.constituents))

    def __repr__(self):
        return "%s => %s [%s]" % (self.terminals[0], self.terminals[1], ", ".join(map(str, self.constituents)))


class split_error(Exception):

    def __init__(self, ident):
        self.ident = ident


class inest_dag(dag):

    def __init__(self, rap_dual, ispace):
        super(inest_dag, self).__init__()
        self.vtype = inest_vertex
        self.etype = inest_edge
        self.source_dag = rap_dual
        self.ispace = ispace
        self.loop_order = list(reversed(ispace.loop_order))  # Why does this have to be reversed?

        # Merge raps with spatial relationships into rapsets.
        rapsets = []
        for rap in rap_dual.vertices.values():
            merged = False
            # TODO maybe have this not be so expensive
            for rapset in rapsets:
                distances = [rap.distance(candidate) for candidate in rapset]
                if all(d is not None for d in distances):
                    rapset |= set([rap])
                    merged = True
                    break
                elif any(d is not None for d in distances):
                    logger.error("Distance between %s and %s not uniform -- this is unexpected." % (rap, rapset))
            if not merged:
                newset = set([rap])
                rapsets += [newset]
        # Construct a new inest for each rapset.
        rapinest = {}
        for rapset in rapsets:
            # Construct list of external/internal variables for each inest.
            # TODO: Pythonify this?
            ex_idents = set([])
            in_idents = set([])
            for rap in set(rapset):
                ex_idents |= rap.outer_ivars()
                in_idents |= rap.inner_ivars()
            assert(ex_idents & in_idents == set([]))
            assert(ex_idents | in_idents <= set(ispace.loop_dict.keys()))

            iv = inest_vertex(inest.make_nest(ex_idents, in_idents, ispace, rapset))
            self.add_vertex(iv)
            for rap in rapset:
                rapinest[rap] = iv.key()

        # Connect inests with edges based on rap_dual dependency information.
        for (s, e), ed in rap_dual.edges.items():
            ie = self.etype(rapinest[s], rapinest[e], [ed])
            if ie.key() not in self.edges:
                self.add_edge(ie)
            else:
                self.edges[ie.key()].constituents += [ed]

    def move_vertex(self, vkorig, inew):
        if inew is not None:
            vnew = self.vtype(inew)
        else:
            vnew = None
        super(inest_dag, self).move_vertex(vkorig, vnew)

    def union(self, A, B):
        if A:
            if B:
                return self.fuse_core(A, B)
            return A
        if B:
            return B
        return None

    def fuse_core(self, A, B):
        # TODO: make sure we don't fuse with incompatible bounds
        if isinstance(A, inest_leaf):
            if isinstance(B, inest_leaf):
                return inest_leaf(A.rapset.union(B.rapset))
            else:
                # ante is a leaf, cons is not
                (Ao, Bo) = (10000, self.loop_order.index(B.iident))  # in theory, 10000 is not as big as it should be
        elif isinstance(B, inest_leaf):
            # cons is a leaf, ante is not
            (Ao, Bo) = (self.loop_order.index(A.iident), 10000)
        else:
            (Ao, Bo) = [self.loop_order.index(x.iident) for x in (A, B)]
        iident = self.loop_order[min(Ao, Bo)]
        if Ao == Bo:
            if (not A.internal and not B.internal and
                 self.source_dag.vertex_group_le(A.phases.prologue_prime(), B.phases[1].child_rapset()) and
                 self.source_dag.vertex_group_le(B.phases.prologue_prime(), A.phases[1].child_rapset()) and
                 self.source_dag.vertex_group_le(A.phases[1].child_rapset(), B.phases.epilogue_prime()) and
                 self.source_dag.vertex_group_le(B.phases[1].child_rapset(), A.phases.epilogue_prime())):
                return inest(iident, False, self.ispace.interval(iident), (self.union(a, b) for (a, b) in it.izip_longest(A.phases, B.phases)))
            else:
                raise split_error(iident)
        else:
            if Ao < Bo:
                (A, B) = (B, A)
            before = self.source_dag.vertex_group_le(A.child_rapset(), B.phases[1].child_rapset())
            after = self.source_dag.vertex_group_le(B.phases[2].child_rapset(), A.child_rapset())
            if before and not A.internal:
                if after:
                    logger.warning("Warning: ambiguous fusion order; picking before!")
                return inest(iident, False, self.ispace.interval(iident), [self.union(A, B.phases[0]), B.phases[1], B.phases[2]])
            elif after and not B.internal:
                return inest(iident, False, self.ispace.interval(iident), [B.phases[0], B.phases[1], self.union(A, B.phases[2])])
            else:
                raise split_error(iident)

    def fuse(self, Ak, Bk):
        z = self.fuse_core(self.vertices[Ak].inest, self.vertices[Bk].inest)
        return self.merge_vertices((Ak, Bk), lambda x: self.vtype(z), lambda k, v: inest_super_edge(k[0], k[1], v))

    def topo_fuse(self):
        split_filt = lambda e: not self.edges[e].is_split()
        topo = self.topo_sort()
        for v in topo:
            current = v
            pe = next(self.incoming_edges(current, split_filt), None)
            while pe:
                ev = self.edges[pe]
                try:
                    current = self.fuse(current, ev.terminals[0])
                except split_error as se:
                    seps = self.separating_edges([ev.terminals[0]], [current])
                    for e in seps:
                        self.edges[e.key()].split(se.ident)
                pe = next(self.incoming_edges(current, split_filt), None)

    def check(self):
        super(inest_dag, self).check()
        for e in self.edges.values():
            assert isinstance(e, self.etype) or isinstance(e, inest_super_edge)

    def rap_fuse(self):
        for inest in [self.vertices[v].inest for v in self.topo_sort()]:
            inest.rap_fuse()

    def gen_loops(self):
        topo = self.topo_sort()
        res = []
        for v in topo:
            res.append(self.vertices[v].inest.gen_loops(self.source_dag, 0))
        return "\n".join(res)
