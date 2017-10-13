# hfav/dag.py; directed acyclic graph datastructure/tools

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
from collections import defaultdict
import operator as op
import logging
import codegen
import re
import heapq
import copy
logger = logging.getLogger(__name__)


class vertex(object):

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "%s" % (self.name,)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def key(self):
        return self.name

class cycleError(Exception):
    def __init__(self, edges):
        self.edges = edges

class edge(object):

    def __init__(self, s, e):
        self.terminals = (s, e)

    def __repr__(self):
        return "from %s to %s" % (self.terminals[0], self.terminals[1])

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def key(self):
        return self.terminals


class super_vertex(vertex):

    def __init__(self, constituents):
        super(super_vertex, self).__init__(tuple(sorted([v.key() for v in constituents])))
        self.constituents = constituents


class super_edge(edge):

    def __init__(self, s, e, constituents):
        super(super_edge, self).__init__(s, e)
        self.constituents = constituents

    def __repr__(self):
        return "from %s to %s [%s]" % (self.terminals[0], self.terminals[1], ", ".join(map(str, self.constituents)))


def iter_len(iter):
    return sum(1 for x in iter)


def dot_san(s):
    return s.replace("\"", "\\\"")


def default_vmerger(verts):
    return super_vertex(verts)


def default_emerger(k, v):
    return super_edge(k[0], k[1], v)


def set_iter(s):
    for x in s:
        yield x
    raise StopIteration()


class dag(object):

    def __init__(self):
        self.vertices = {}
        self.outgoing = {}
        self.incoming = {}
        self.edges = {}
        self.etype = edge
        self.vtype = vertex

    @classmethod
    def from_edgelist(cls, el):
        d = cls()
        for s, e in el:
            d.add_edge_verts(d.etype(s, e))
        return d

    def subgraph(self, vertices):
        d = self.__class__()
        for vk in vertices:
            d.add_vertex(vk)
        d.edges = {}
        for e in self.edges.values():
            if all(k in d.vertices for k in e.terminals):
                d.add_edge(e)
        return d

    def empty(self):
        return self.vertices == {} and self.edges == {}

    def merge_vertices(self, vkeys, vmerger=default_vmerger, emerger=default_emerger):
        old_edges = self.edges.copy()
        verts = []
        for vk in vkeys:
            verts.append(self.vertices[vk])
            self.remove_vertex(vk)
        nv = vmerger(verts)
        self.add_vertex(nv)
        new_edges = defaultdict(list)
        for s, e in old_edges.keys():
            if s in vkeys:
                if not e in vkeys:
                    new_edges[(nv.key(), e)].append(old_edges[(s, e)])
                if (s, e) in self.edges:
                    self.remove_edge((s, e))
            elif e in vkeys:
                if (s, nv.key()) not in self.edges:
                    new_edges[(s, nv.key())].append(old_edges[(s, e)])
                if (s, e) in self.edges:
                    self.remove_edge((s, e))
        for k, v in new_edges.items():
            self.add_edge(emerger(k, v))
        return nv.key()

    def copy(self):
        c = copy.copy(self)
        c.vertices = self.vertices.copy()
        c.incoming = dict((k, v.copy()) for k, v in self.incoming.items())
        c.outgoing = dict((k, v.copy()) for k, v in self.outgoing.items())
        c.edges = self.edges.copy()
        return c

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def add_vertex(self, v):
        if v.key() in self.vertices:
            raise ValueError("already have vertex %s in DAG" % (v.key(),))
        self.vertices[v.key()] = v
        if v.key() not in self.incoming:
            self.incoming[v.key()] = set()
        if v.key() not in self.outgoing:
            self.outgoing[v.key()] = set()
        return v

    def remove_edge(self, ek):
        e = self.edges[ek]
        self.outgoing[e.terminals[0]].remove(ek)
        self.incoming[e.terminals[1]].remove(ek)
        del self.edges[ek]

    def remove_vertex(self, vk):
        for ek in self.outgoing_edges(vk):
            self.remove_edge(ek)
        for ek in self.incoming_edges(vk):
            self.remove_edge(ek)
        del self.vertices[vk]
        del self.outgoing[vk]
        del self.incoming[vk]

    def move_vertex(self, vko, vn):
        ec = self.edges.items()[:]
        if vn is not None:
            self.remove_vertex(vko)
            self.add_vertex(vn)
            # TODO: Why is it this?
            #if vko == vn.key():
            #    return
        else:
            self.remove_vertex(vko)

        for (es, ee), ev in ec:
            if es == vko:
                if (es, ee) in self.edges:
                    self.remove_edge((es, ee))
                if vn is not None:
                    ev.terminals = (vn.key(), ee)
                    self.add_edge(ev)
            elif ee == vko:
                if (es, ee) in self.edges:
                    self.remove_edge((es, ee))
                if vn is not None:
                    ev.terminals = (es, vn.key())
                    self.add_edge(ev)

    def add_edge(self, e):
        if e.key() in self.edges:
            raise ValueError("already have edge %s in DAG" % (e.key(),))
        if not e.terminals[0] in self.vertices:
            raise ValueError("vertex %s at start of edge %s not in DAG" % (e.terminals[0], e.key(),))
        if not e.terminals[1] in self.vertices:
            raise ValueError("vertex %s at end of edge %s not in DAG" % (e.terminals[1], e.key(),))
        self.edges[e.key()] = e
        self.outgoing[e.terminals[0]].add(e.key())
        self.incoming[e.terminals[1]].add(e.key())
        return e

    def add_edge_spec(self, e):
        if e.key() in self.edges:
            raise ValueError("already have edge %s in DAG" % (e.key(),))
        self.edges[e.key()] = e
        if not e.terminals[0] in self.outgoing:
            self.outgoing[e.terminals[0]] = set()
            # assert not e.terminals[0] in self.incoming
            # self.incoming[e.terminals[0]] = set()
        self.outgoing[e.terminals[0]].add(e.key())
        if not e.terminals[1] in self.incoming:
            self.incoming[e.terminals[1]] = set()
            # assert not e.terminals[1] in self.outgoing
            # self.outgoing[e.terminals[1]] = set()
        self.incoming[e.terminals[1]].add(e.key())
        return e

    def add_edge_verts(self, e):
        if e.key() in self.edges:
            raise ValueError("already have edge %s in DAG" % (e.key(),))
        if not e.terminals[0] in self.vertices:
            self.add_vertex(self.vtype(e.terminals[0]))
        if not e.terminals[1] in self.vertices:
            self.add_vertex(self.vtype(e.terminals[1]))
        self.edges[e.key()] = e
        self.outgoing[e.terminals[0]].add(e.key())
        self.incoming[e.terminals[1]].add(e.key())
        return e

    def check(self):
        for (s, e) in self.edges.keys():
            assert s in self.vertices
            assert (s, e) in self.incoming[e]
            assert (s, e) in self.outgoing_edges(s)
            assert e in self.vertices
            assert (s, e) in self.outgoing[s]
            assert (s, e) in self.incoming_edges(e)
        itally = 0
        otally = 0
        for v in self.vertices.keys():
            itally += self.idegree(v)
            otally += self.odegree(v)
            assert v in self.outgoing
            assert v in self.incoming
        assert itally == otally
        assert itally == len(self.edges)
        for v, oes in self.outgoing.items():
            for oe in oes:
                assert v == oe[0]
                assert oe in self.edges
        for v, ies in self.incoming.items():
            for ie in ies:
                assert v == ie[1]
                assert ie in self.edges
        assert self.cycle_free()
        return True

    def idegree(self, vkey):
        return len(self.incoming[vkey])

    def odegree(self, vkey):
        return len(self.outgoing[vkey])

    def incoming_edges(self, vkey, filt=lambda e: True):
        # TODO: use iterators where possible
        return it.ifilter(filt, iter(list(self.incoming[vkey])))

    def outgoing_edges(self, vkey, filt=lambda e: True):
        return it.ifilter(filt, iter(list(self.outgoing[vkey])))

    def incoming_vertices(self, vkey):
        return iter((e[0] for e in set_iter(self.incoming_edges(vkey))))

    def outgoing_vertices(self, vkey):
        return iter((e[1] for e in set_iter(self.outgoing_edges(vkey))))

    def roots(self):
        return it.ifilter(lambda v: self.idegree(v) == 0, self.vertices.keys())

    def leaves(self):
        return it.ifilter(lambda v: self.odegree(v) == 0, self.vertices.keys())

    def topo_sort(self, priority=lambda x: 0):
        order = []
        action = lambda x: order.append(x.key())
        self.topo_visit(action=action, priority=priority)
        return order

    def topo_visit(self, action, priority=lambda x: 0):
        dp = self.copy()
        roots = list((priority(v), v) for v in dp.roots())
        heapq.heapify(roots)
        while len(roots) > 0:
            minitem = heapq.heappop(roots)
            c = dp.vertices[minitem[1]]
            action(c)
            outg = dp.outgoing_edges(c.key())
            for o in outg:
                child = dp.vertices[o[1]]
                dp.remove_edge(o)
                if dp.idegree(child.key()) == 0:
                    heapq.heappush(roots, (priority(child.key()), child.key()))
        if len(dp.edges) != 0:
            raise cycleError(dp.edges)

    def cycle_free(self):
        try:
            self.topo_sort()
            return True
        except cycleError:
            return False

    def reverse(self):
        new_edges = {}
        for e in self.edges.values():
            e.terminals = (e.terminals[1], e.terminals[0])
            new_edges[e.key()] = e
        self.edges = new_edges

    def level_sort_levels(self):
        max_depths = self.vertex_max_depths()
        decreasing_v = sorted(max_depths.items(), key=op.itemgetter(1), reverse=True)
        return [[v[0] for v in l] for (d, l) in it.groupby(decreasing_v, key=op.itemgetter(1))]

    def level_sort(self, keyf=lambda x: x):
        ls = self.level_sort_levels()
        order = []
        max_depth = len(ls) - 1
        for depth in range(max_depth, -1, -1):
            order += sorted(ls[depth], key=keyf)
        return order

    def vertex_max_depths(self):
        torder = self.topo_sort()
        for v in torder:
            assert v in self.vertices
        length_ending = defaultdict(int)
        for v in torder:
            if not v in length_ending:
                length_ending[v] = 0
            for inc in self.incoming_vertices(v):
                length_ending[v] = max(length_ending[v], length_ending[inc] + 1)
        return length_ending

    def longest_path(self):
        max_depths = self.vertex_max_depths()
        decreasing_v = [k for (k, v) in sorted(max_depths.items(), key=op.itemgetter(1), reverse=True)]
        paths = []
        while len(decreasing_v) > 0:
            path = [decreasing_v.pop(0)]
            current = path[0]
            while len(decreasing_v) > 0:
                inc = list(it.ifilter(lambda x: x[0] in decreasing_v, self.incoming_edges(current)))
                if len(inc) == 0:
                    break
                current = max(((s, max_depths[s]) for (s, e) in inc), key=op.itemgetter(1))[0]
                path.append(current)
                decreasing_v.remove(current)
            if len(path) > 1:
                paths.append(path)
        return paths

    def longest_path_components(self):
        lp = self.longest_path()
        seen = []
        comps = []
        for p in lp:
            pd = dag.from_edgelist(it.izip(p[1:], p))
            for v in pd.vertices.keys():
                assert not v in seen
                seen += [v]
            comps += [pd]
        for v in self.vertices.keys():
            if v not in seen:
                di = dag()
                di.add_vertex(self.vtype(v))
                comps += [di]
        return comps

    def stats(self):
        nv = len(self.vertices)
        ne = len(self.edges)
        nr = len(list(self.roots()))
        nl = len(list(self.leaves()))
        # print self.dot()
        lp = self.longest_path()
        assert lp
        cp = len(lp[0])
        return "%d vertices (%d roots, %d leaves) %s edges, critical path len: %s" % (nv, nr, nl, ne, cp)

    def connected_components(self):
        """ Pick a root in DAG, set as current vertex
            Put all edges from v into equeue
            Move current vertex from source DAG to current connected component
            For each edge in equeue:
                pick other vertex, make current vertex, move edge
                move current vertex
            Current dag is complete, repeat for next as long as there are roots
        """
        # TODO: have this use new add/remove code
        dp = self.copy()
        comps = []
        while True:
            try:
                rootk = next(dp.roots())
            except StopIteration:
                return comps
            cls = self.__class__
            comp = cls.__new__(cls)

            equeue = [('o', dp.edges[x]) for x in dp.outgoing_edges(rootk)] + [('i', dp.edges[x]) for x in dp.incoming_edges(rootk)]
            root = dp.vertices[rootk]
            dp.remove_vertex(rootk)
            comp.add_vertex(root)
            while len(equeue) > 0:
                (d, e) = equeue.pop()
                if e.key() in comp.edges:
                    continue
                if d == 'o':
                    vk = e.terminals[1]
                elif d == 'i':
                    vk = e.terminals[0]
                edel = [('o', self.edges[x]) for x in dp.outgoing_edges(vk)] + [('i', self.edges[x]) for x in dp.incoming_edges(vk)]
                equeue += edel
                if vk in dp.vertices.keys():
                    comp.add_vertex(dp.vertices[vk])
                    dp.remove_vertex(vk)
                else:
                    assert vk in comp.vertices.keys()
                comp.add_edge(e)
            comps.append(comp)

    def dot(self, v_fmt=str, e_fmt=str):
        lst = codegen.listing()
        lst.append("digraph\n")
        lst.append("{\n")
        lst.append("\n")
        lst.indent()
        lst.append("ratio=fill;\n")
        lst.append("node [shape=box];\n")
        lst.append("\n")
        rx = re.compile("[!?\])]")
        vn = dict((vk, d) for (d, vk) in enumerate(self.vertices.keys()))
        for ((s, e), ev) in self.edges.items():
            lst.append("%d -> %d [label=\"%s\"];\n" % (vn[s], vn[e], dot_san(e_fmt(ev))))
        lst.append("\n")
        for vk, vv in self.vertices.items():
            lst.append("%d [label=\"%s\"];\n" % (vn[vk], dot_san(v_fmt(vv))))
        lst.append("\n")
        lst.deindent()
        lst.append("}\n")
        return lst.emit()

    def reachable(self, start, end):
        if start == end:
            return True
        return any(self.reachable(o, end) for o in self.outgoing_vertices(start))

    def vertex_order(self, group1, group2):
        """Inputs are lists of vertices.
           Return order of inputs as either true (group1 can reach group2), false (group2 can reach group1),
           or None (not reachable from the other). Throws an exception if there is a cycle."""
        c = self.copy()
        m1 = c.merge_vertices(group1)
        m2 = c.merge_vertices(group2)
        g1tog2 = c.reachable(m1, m2)
        g2tog1 = c.reachable(m2, m1)
        if not g1tog2 and not g2tog1:
            return None
        if g1tog2 and g2tog1:
            raise ValueError("Unorderable")
        return g1tog2

    def vertex_group_lt(self, group1, group2):
        """Group1 and group2 are iterables of vertices. Return if the first group can reach the second"""
        c = self.copy()
        m1 = c.merge_vertices(group1)
        m2 = c.merge_vertices(group2)
        return c.reachable(m1, m2)

    def vertex_group_le(self, group1, group2):
        """Group1 and group2 are iterables of vertices. Return if the second group can NOT reach the second"""
        c = self.copy()
        m1 = c.merge_vertices(group1)
        m2 = c.merge_vertices(group2)
        return not c.reachable(m2, m1)

    def descendents(self, v):
        res = []
        verts = [v]
        while len(verts) > 0:
            c = verts.pop(0)
            res.append(c)
            verts += list(self.outgoing_vertices(c))
        return res

    def ancestors(self, v):
        res = []
        verts = [v]
        while len(verts) > 0:
            c = verts.pop(0)
            res.append(c)
            verts += list(self.incoming_vertices(c))
        return res

    def separating_edges(self, ante, cons):
        """This returns a (minimal) set of edges that separates all ancestors to ante from all descendents of cons.
        There are usually many such sets of edges that can do this, and this currently just returns all outgoing from the supernode of the ancestors."""
        cpy = self.copy()
        #after = cpy.merge_vertices(set(*[cpy.descendents(v) for v in cons]))
        before = cpy.merge_vertices(set(*[cpy.ancestors(v) for v in ante]))
        return list(it.chain(*(cpy.edges[x].constituents for x in cpy.outgoing_edges(before))))


if __name__ == '__main__':
    d = dag.from_edgelist([("A", "C"),
                           ("A", "B"),
                           ("B", "C"),
                           ("C", "D"),
                           ("A", "D"),
                           ("D", "E")])
    d.add_vertex(vertex("WW"))
    t = d.topo_sort()
    print t
    l = d.level_sort()
    print l
    lp = d.longest_path()
    print lp

    for v0 in d.vertices:
        for v1 in d.vertices:
            print v0, v1, d.reachable(v0, v1)

    print d.vertex_order(["A", "B"], ["WW"])
    try:
        print d.vertex_order(["A", "C"], ["B", "E"])
        print "This should have been unorderable"
    except ValueError:
        print "Unorderable, as expected"
    print d.vertex_order(["A", "B"], ["C", "D"])
    print d.vertex_order(["C", "D"], ["A", "B"])

    print d.vertex_group_le(["C", "D"], ["A", "B"])

    dp = d.subgraph(["A", "D"])
    t = dp.topo_sort()
    print t
    l = dp.level_sort()
    print l
    lp = dp.longest_path()
    print lp

    for v0 in dp.vertices:
        for v1 in dp.vertices:
            print v0, v1, dp.reachable(v0, v1)

    d.merge_vertices(["B", "C"])
    t = d.topo_sort()
    print t
    l = d.level_sort()
    print l
    lp = d.longest_path()
    print lp

    for v0 in d.vertices:
        for v1 in d.vertices:
            print v0, v1, d.reachable(v0, v1)
