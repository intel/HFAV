# hfav/infer.py; DAG inferfence code

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
import numbers
import sys
import copy
import collections
import logging
import operator as op
from . import parse
from . import term
from . import codegen

from dag import dag, edge
import re

logger = logging.getLogger(__name__)


class production(object):

    def __init__(self, antecedent=[], consequent=[]):
        assert not (set(antecedent) & set(consequent)) # TODO: How bad an idea is this...
        self.antecedent = antecedent
        self.consequent = consequent

    def unify(self, res):
        """Try to unify res with each consequent; return ones that match"""
        unified = []
        for o in self.consequent:
            u = res.unify(o)
            if u is not None:
                unified.append(u)
        if len(unified) == 1:
            return unified[0]
        elif len(unified) == 0:
            return None
        elif self.__class__.__name__ == "codeblock": # Sorry
            # Check to see if one of the unifications is a substitution with no offsets
            # TODO: Worry about the implications of this later; for now, boundary conditions should be fine
            for u in unified:
                if all([subs == var.as_symbolic_constant() for (var, subs) in u.items()]):
                    return u
            assert 0
        else:
            assert 0


class rule_arg(object):

    def __init__(self, position, type, arg, io):  # hack
        self.position = position
        self.type = type
        self.arg = parse.parser(arg).expr().canonize()
        self.io = io

    def __str__(self):
        return str(self.arg)

    def __repr__(self):
        return "%s(%r, %s, %r, %r)" % (self.__class__.__name__, self.position, self.type, self.arg, self.io)

    @classmethod
    def from_substitute(cls, rule_arg, subs):
        res = copy.deepcopy(rule_arg)
        res.arg = rule_arg.arg.substitute(subs).canonize()
        return res


class unused(production):

    def __init__(self):
        super(unused, self).__init__([], [])
        self.name = "__unused__"
        self.args = []

    def get_name(self):
        return self.name


class rule(production):

    def __init__(self, name, args=[], antecedent=[], consequent=[]):
        super(rule, self).__init__(antecedent, consequent)
        assert(len(antecedent) >= 0)
        assert(len(consequent) > 0)
        self.name = name
        self.args = args

    def __repr__(self):
        return "%s(%s, %r, %r, %r)" % (self.__class__.__name__, self.name, self.args, self.antecedent, self.consequent)

    def get_name(self):
        return self.name

    @classmethod
    def read(cls, name, iargs, oargs):  # renamed to make it clear it reads something different to the others
        args = sorted(iargs + oargs, key=lambda x: x.position)
        ante = [z.arg for z in iargs]
        cons = [z.arg for z in oargs]
        return rule(name, args, ante, cons)


class axiom(production):

    def __init__(self, antecedent, consequent, type):
        super(axiom, self).__init__([antecedent], [consequent])
        self.type = type

    def __repr__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.antecedent[0], self.consequent[0], self.type)

    def get_name(self):
        return 'axiom'

    @classmethod
    def read(cls, antecedent, consequent, type):
        ante = term.function("_inref", [parse.parser(antecedent).expr().canonize()])
        cons = parse.parser(consequent).expr().canonize()
        return axiom(ante, cons, type)

    def is_in(self, term):
        prod = self.consequent[0].unify(term)
        if prod is not None:
            return load_app(self, prod)
        return None


class ivar_axiom(axiom):

    def __init__(self, ivar):
        super(ivar_axiom, self).__init__(term.function("_inref", [ivar]), ivar, 'int')

    def get_name(self):
        return 'ivar_axiom'

    def is_in(self, term):
        """ Note that this is sort of hacky in that it will accept only expressions that have just this variable's symbolic value. This will probably accept things it shouldn't (like foo(i[0][0]), and will not allow things it should (i + j)"""
        tsc = term.symbolic_constants()
        if len(tsc) == 1 and tsc[0] == self.consequent[0]:
            return ivar_app(self)
        return None


class reduction_op(object):

    def __init__(self, prefix, identity, fstring):
        self.prefix = prefix
        self.identity = identity
        self.fstring = fstring

    @classmethod
    def supported(cls):
        return {"+": reduction_op("plus", "0", "%s + %s"), "max": reduction_op("max", "INT_MIN", "__hfav_max(%s, %s)"), "|": reduction_op("or", "false", "%s | %s")}


class reduction_initializer(production):

    def __init__(self, var, op, type):
        super(reduction_initializer, self).__init__([], [term.function("_init", [var])])
        self.op = op
        self.args = [rule_arg(-1, type, str(self.consequent[0]), "output")]

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.consequent[0])

    def get_name(self):
        return 'reduction_initializer'


class reduction_finalizer(production):

    def __init__(self, var, op, type):
        super(reduction_finalizer, self).__init__([term.function("_reduction", [var])], [var])
        self.op = op
        self.args = [rule_arg(-1, type, str(self.consequent[0]), "output"), rule_arg(0, type, str(self.antecedent[0]), "input")]

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.antecedent[0], self.consequent[0])

    def get_name(self):
        return 'reduction_finalizer'


class goal(production):

    def __init__(self, antecedent, consequent, type):
        super(goal, self).__init__(antecedent, consequent)
        self.type = type

    def __repr__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.antecedent, self.consequent, self.type)

    def get_name(self):
        return 'goal'

    @classmethod
    def read(cls, antecedent, consequent, type):
        ante = parse.parser(antecedent).expr().canonize()
        cons = parse.parser(consequent).expr().canonize()
        # seems nuts, but there's a reason! make the cons an antecedent so we don't overwrite.
        return goal([ante, cons], [term.function("_outref", [cons])], type)


class chain_failed(Exception):
    pass


def group_vars(v):
    return it.groupby(sorted(v, key=var_key_noarray), var_key_noarray)


class codeblock(rule):

    """A codeblock is a representation of arbitrary user-code; this is encoded as a set of production rules"""

    def __init__(self, name, code, args, antecedent, consequent, type, outer, inner):
        super(codeblock, self).__init__(name, args, antecedent, consequent)
        self.code = code
        self.outer_iv = outer
        self.inner_iv = inner

    @classmethod
    def parse_var(cls, string):
        end = 0
        refs = []
        start = string.find("[", end)
        ident = string[:start]
        while True:
            if start == -1:
                break
                #return ident, refs
            end = string.find("]", start)
            if end == -1:
                raise SyntaxError("Unclosed '[' in codeblock definition")
            refs.append(string[start+1:end])
            start = string.find("[", end)

        base_str = ident
        offsets = list(refs)
        for (d, r) in enumerate(refs):
            if '*' in r:
                offsets[d] = [0]
                idx = r.replace('*', '?')
            elif ':' in r:
                o = r.split(":")
                idx = o[0].split('?')[0] + "?"
                left = int(o[0].split('?')[1])
                right = int(o[1]) + 1
                offsets[d] = range(left, right)
            else:
                offsets[d] = [0]
                idx = r
            base_str += "[" + idx + "]"

        # Hack
        if base_str.count('(') > base_str.count(')'):
            base_str += ')' * (base_str.count('(') - base_str.count(')'))

        offsets = list(reversed(offsets))
        refs = list(reversed(refs))

        base = parse.parser(base_str).expr().canonize()

        all_ivars = [base.position_tuple()[d] for d in range(len(refs))]
        inner_ivars = [base.position_tuple()[d] for d in range(len(refs)) if '*' in refs[d]]

        terms = []
        for position in it.product(*offsets):
            subs = dict((var, term.add([var, term.numeric_constant(position[p])]).canonize()) for (p, var) in enumerate(base.position_tuple()))
            terms.append(base.substitute(subs))

        return base_str, terms, all_ivars, inner_ivars

    @classmethod
    def read(cls, name, cb):

        args = []
        ante = []
        cons = []

        all_ivars = set([])
        outer_ivars = set([])
        inner_ivars = set([])

        code = cb["code"]

        for line in cb["inputs"].splitlines():
            input = line.partition("=>")
            if input[1] == "=>":
                assert 0
            else:
                t, decl = input[0].split()
                base_str, terms, all_iv, inner_iv = codeblock.parse_var(decl)
                args += [rule_arg(0, t, base_str, "input")]
                ante += terms
                all_ivars |= set(all_iv)
                inner_ivars |= set(inner_iv)

        for line in cb["outputs"].splitlines():
            output = line.partition("=>")
            if output[1] == "=>":
                assert 0
            else:
                t, decl = output[0].split()
                base_str, terms, all_iv, inner_iv = codeblock.parse_var(decl)
                args += [rule_arg(0, t, base_str, "output")]
                cons += terms
                all_ivars |= set(all_iv)
                inner_ivars |= set(inner_iv)

        argc = 0
        for (var, group) in it.groupby(args, lambda x: var_key_noarray(x.arg)):
            for arg in group:
                arg.position = argc
            argc += 1

        for (i,arg) in enumerate(args):
            print i, args[i], args[i].position, args[i].io

        outer_ivars = all_ivars - inner_ivars

        return codeblock(name, code, args, ante, cons, t, set(x.as_symbolic_constant() for x in outer_ivars), set(x.as_symbolic_constant() for x in inner_ivars))


class vertex(object):

    """Vertices represent terms, and edges point to rule applications (many edges can comprise such an application)"""

    def __init__(self, term, type="int"):
        self.term = term
        self.type = type

    def __repr__(self):
        return repr(self.term)

    def __str__(self):
        return str(self.term)

    def key(self):
        return self.term

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


class inference_dag(dag):

    def __init__(self, typedict):
        dag.__init__(self)
        self.typedict = typedict
        self.vtype = vertex

    def has_resolved_term(self, term):
        return term in self.vertices and self.vertices[term].resolved

    def internal_vertices(self):
        return it.ifilter(lambda x: self.idegree(x.key()) > 0, self.vertices.values())  # SJP: I think we should care about things with odegree == 0; we may need to write them out.
        # return it.ifilter(lambda x: x.idegree() > 0 and x.odegree() > 0, self.vertices.values())

    def constant_vars(self):
        cdag = self.aggregate_dag()
        input_vars = set(x.replace("_inref_", "") for x in cdag.roots())
        output_vars = set(x.replace("_outref_", "") for x in cdag.leaves())
        return (input_vars - output_vars) | set(["null"])

    def connected_components(self):
        """ Return a list of connected components ordered by split links"""

        comps = super(inference_dag, self).connected_components()

        # Build a dag representing the relationship between the connected components.
        comp_dag = dag()
        for d in range(0, len(comps)):
            comp_dag.add_vertex(self.vtype(d))
        isplit = [set(var_key_noarray(v).replace("_inref__split_", "") for v in comps[d].roots() if "_split_" in var_key_noarray(v)) for d in range(0, len(comps))]
        osplit = [set(var_key_noarray(v).replace("_outref__split_", "") for v in comps[d].leaves() if "_split_" in var_key_noarray(v)) for d in range(0, len(comps))]
        for d1 in range(0, len(comps)):
            for d2 in range(d1 + 1, len(comps)):
                if len(osplit[d1] & isplit[d2]) > 0:
                    comp_dag.add_edge(edge(d1, d2))
                if len(osplit[d2] & isplit[d1]) > 0:
                    comp_dag.add_edge(edge(d2, d1))

        # Order the components based on a topological sort of the component dag.
        return [comps[d] for d in comp_dag.topo_sort()]

    def stats(self):
        return super(inference_dag, self).stats() + " RAPs: %s" % (len(list(self.raps())),)

    def aggregate_dag(self):
        assert(False)
        res = dag()
        for (n, l) in group_vars([v.term for v in self.vertices.values()]):
            nv = self.vtype(n)
            nv.pos = list(l)
            res.add_vertex(nv)
        for ((s, e), ev) in self.edges.items():
            ne = edge(var_key_noarray(s), var_key_noarray(e))
            ne.raps = [ev.rap]
            if ne.key() not in res.edges:
                res.add_edge(ne)
            else:
                res.edges[ne.key()].raps += ne.raps
        return res

    def internal_aggregate_dag(self):
        res = dag()
        for (n, l) in group_vars([v.term for v in self.internal_vertices()]):
            nv = self.vtype(n)
            nv.pos = list(l)
            res.add_vertex(nv)
        for ((s, e), ev) in self.edges.items():
            if not var_key_noarray(s) in res.vertices or not var_key_noarray(e) in res.vertices:
                continue
            ne = edge(var_key_noarray(s), var_key_noarray(e))
            ne.raps = [ev.rap]
            if ne.key() not in res.edges:
                res.add_edge(ne)
            else:
                res.edges[ne.key()].raps += ne.raps
        return res

    def ante_app_counts(self):
        ante_app_counts = collections.defaultdict(int)
        for k, v in self.vertices.items():
            # TODO: this is probably less efficient than it ought to be
            for oe in self.outgoing_edges(k):
                ante_app_counts[self.edges[oe].rap] += 1
        return ante_app_counts

    def rule_apply(self, rap):
        if rap.antecedent == []:
            ante = term.atom()
            ante.ident = "null"
            v = self.add_vertex(ante, None)
            for cons in rap.consequent:
                v = self.resolve_vertex(cons, None)
            for cons in rap.consequent:
                ne = edge(ante, cons)
                ne.rap = rap
                self.add_edge(ne)
        else:
            for ante in rap.antecedent:
                v = self.add_vertex(ante, rap.get_type(ante))
            for cons in rap.consequent:
                v = self.resolve_vertex(cons, rap.get_type(cons))
            for ante in set(rap.antecedent):  # A rap that uses the same input twice should only generate one edge
                for cons in rap.consequent:
                    ne = edge(ante, cons)
                    ne.rap = rap
                    self.add_edge(ne)

    def raps(self):
        seen = set()
        for e in self.edges.values():
            if e.rap not in seen:
                seen.add(e.rap)
                yield e.rap
        raise StopIteration()

    def ivars(self):
        s = set()
        s.update(*(x.rap_ivars() for x in self.raps()))
        return s

    def dot(self):
        return super(inference_dag, self).dot(v_fmt=lambda x: str(x.key()), e_fmt=lambda x: x.rap.name())

    def add_vertex(self, term, type):
        if term not in self.vertices:
            super(inference_dag, self).add_vertex(self.vtype(term, type))
            self.vertices[term].resolved = False
        elif self.vertices[term].type is None or type is None:
            pass  # A "None" can be stored in anything
        elif self.typedict[self.vertices[term].type] != self.typedict[type]:
            raise TypeError("Expected %s of type %s (%s%s), found %s (%s%s)" % (term, type, self.typedict[type][0], self.typedict[type][
                            1], self.vertices[term].type, self.typedict[self.vertices[term].type][0], self.typedict[self.vertices[term].type][1]))
        return self.vertices[term]

    def resolve_vertex(self, term, type):
        if term not in self.vertices:
            super(inference_dag, self).add_vertex(self.vtype(term, type))
        elif self.vertices[term].type is None or type is None:
            pass  # A "None" can be stored in anything
        elif self.typedict[self.vertices[term].type] != self.typedict[type]:
            raise TypeError("Expected %s of type %s (%s%s), found %s (%s%s)" % (term, type, self.typedict[type][0], self.typedict[type][
                            1], self.vertices[term].type, self.typedict[self.vertices[term].type][0], self.typedict[self.vertices[term].type][1]))
        self.vertices[term].resolved = True
        return self.vertices[term]


def var_key_noarray(v):
    """Canonize an expression with 'at' references
    TODO: fold into term classes"""
    if isinstance(v, term.at) or isinstance(v, term.add) or isinstance(v, term.neg):
        return var_key_noarray(v.args[0])
    elif isinstance(v, term.function):
        return v.ident + '_' + '_'.join((var_key_noarray(z) for z in v.args))
    return v.name().rstrip("?!")


class rule_app_base(object):

    def __init__(self, rule, symb):
        self.rule = rule
        self.sub = symb
        self.antecedent = [x.substitute(self.sub).canonize() for x in self.rule.antecedent]
        self.consequent = [x.substitute(self.sub).canonize() for x in self.rule.consequent]
        self.types = {}

        self.outer_iv = set(it.chain(*(v.position().variables() for v in self.antecedent + self.consequent)))
        self.inner_iv = set([])

        # TODO: What does this do???
        for a in self.antecedent:
            ad = self.rap_ivars() - a.position().variables()
        for c in self.consequent:
            cd = self.rap_ivars() - c.position().variables()

    def in_zone(self, varset, zone):
        if varset.is_input_variable(self.consequent[0]):
            return False
        active = [varset.is_active(zone, x) for x in self.consequent]
        if any(active) and not all(active):
            logger.debug("Some but not all consequents in %s active in zone %s" % (self.pretty(), zone))
        return any(active)

    def get_type(self, var):
        return self.types[var]

    def terminal(self):
        return True

    def rap_incoming_ivars(self):
        return set(it.chain(*(v.position().variables() for v in self.antecedent)))

    def rap_outgoing_ivars(self):
        return set(it.chain(*(v.position().variables() for v in self.consequent)))

    def outer_ivars(self):
        return set(self.outer_iv)

    def inner_ivars(self):
        return set(self.inner_iv)

    def rap_ivars(self):
        return self.outer_iv | self.inner_iv

    def reduction_ivars(self):
        return self.rap_incoming_ivars() - self.rap_outgoing_ivars()

    def broadcast_ivars(self):
        return self.rap_outgoing_ivars() - self.rap_incoming_ivars()

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.rule, self.sub)

    def __str__(self):
        return self.name() + "(" + ", ".join((str(i) for i in self.antecedent)) + ") => (" + ", ".join((str(i) for i in self.consequent)) + ")"

    def pretty(self):
        return [str(self)]

    def key(self):
        return self

    def endpoint(self):
        return False

    def distance(self, other):
        """Compute spatial distance between raps on an argument-by-argument basis.  Returns None when no spatial relationship exists."""
        if self.rule.get_name() == other.rule.get_name():
            distancedict = collections.defaultdict(int)
            self_args = self.antecedent + self.consequent
            other_args = other.antecedent + other.consequent
            for arg in range(0, len(self_args)):

                self_var = var_key_noarray(self_args[arg])
                self_pos = dict(self_args[arg].position().items)

                other_var = var_key_noarray(other_args[arg])
                other_pos = dict(other_args[arg].position().items)

                # Variable mismatch => not a spatial relationship.
                if self_var != other_var:
                    return None

                # Substitutes have different dimensionality => not a spatial relationship.
                if set(self_pos.keys()) != set(other_pos.keys()):
                    return None

                # Unique non-zero distance for each variable => spatial relationship!
                for pvar in self_pos:
                    dist = other_pos[pvar] - self_pos[pvar]
                    if pvar not in distancedict or distancedict[pvar] == 0:
                        distancedict[pvar] = dist
                    elif dist == 0 or distancedict[pvar] == dist:
                        pass
                    elif distancedict[pvar] != dist:
                        return None
                    else:
                        assert(False)

            return distancedict
        return None


def raplist_distances(raplist):
    if raplist == []:
        return raplist
    raplist = sorted(raplist)
    initial = raplist[0]
    init_dist = []
    other = []
    for rap in raplist:
        d = initial.distance(rap)
        if d:
            init_dist.append((rap, d))
        else:
            other.append(rap)
    return [init_dist] + raplist_distances(other)


def canonical_distances(distlist, loop_order):
    min_item = 0
    min_score = tuple((distlist[min_item][1][ind] for ind in loop_order))
    for c, (r, d) in enumerate(distlist[1:]):
        d_score = tuple((d[ind] for ind in loop_order))
        if d_score < min_score:
            min_item = c + 1
            min_score = tuple((distlist[min_item][1][ind] for ind in loop_order))
    min_item = distlist[min_item][1].copy()
    for (r, d) in distlist:
        for k, v in min_item.iteritems():
            d[k] = d[k] - v
    return sorted(distlist, key=lambda x: tuple((x[1][ind] for ind in loop_order)))


class unused_app(rule_app_base):

    def __init__(self):
        super(unused_app, self).__init__(unused(), {})

    def name(self):
        return self.rule.name

    def emit(self, varset, cg):
        return None

    def in_zone(self, varset, zone):
        return False


class rule_app(rule_app_base):

    """A piece of the inference chain; points to a rule (or an axiom), records substituted antecedents & consequents"""

    def __init__(self, rule, symb):
        super(rule_app, self).__init__(rule, symb)

    def terminal(self):
        return False

    @classmethod
    def from_prod(cls, rule, symb):
        if isinstance(rule, reduction_initializer):
            res = reduction_initializer_app(rule, symb)
        elif isinstance(rule, reduction_finalizer):
            res = reduction_finalizer_app(rule, symb)
        elif isinstance(rule, codeblock):
            res = codeblock_app(rule, symb)
        else:
            res = rule_app(rule, symb)
        res.args = [rule_arg.from_substitute(x, res.sub) for x in res.rule.args]
        # TODO: So much redundant information in this object...
        for a in res.args:
            res.types[a.arg] = a.type
            if isinstance(rule, codeblock):
                res.types[var_key_noarray(a.arg)] = a.type
        res.compute_inouts()
        return res

    def name(self):
        return self.rule.get_name()

    def __lt__(self, other):

        if hasattr(self, 'consequents'):
            if hasattr(other, 'consequents'):
                consequents_lt = all([lambda x, y: x < y for x, y in it.izip_longest(self.consequents, other.consequents)])
            else:
                consequents_lt = True
        else:
            if hasattr(other, 'consequents'):
                consequents_lt = False
            else:
                consequents_lt = True
        if hasattr(self, 'antecedents'):
            if hasattr(other, 'antecedents'):
                antecedents_lt = all([lambda x, y: x < y for x, y in it.izip_longest(self.antecedents, other.antecedents)])
            else:
                antecedents_lt = True
        else:
            if hasattr(other, 'antecedents'):
                antecedents_lt = False
            else:
                antecedents_lt = True

        return self.name() < other.name() and consequents_lt and antecedents_lt

    def split(self, split_vars):
        res = copy.deepcopy(self)
        sf = lambda x, y: term.function(x, [y]) if len(split_vars[var_key_noarray(y)]) > 0 else y
        res.args = [rule_arg(a.position, a.type, str(sf("_post_split", a.arg) if a.io == "input" else sf("_pre_split", a.arg)), a.io) for a in res.args]
        res.types = {}
        for a in res.args:
            res.types[a.arg] = a.type
        res.antecedent = [sf("_post_split", ante) for ante in res.antecedent]
        res.consequent = [sf("_pre_split", cons) for cons in res.consequent]
        return res

    def compute_inouts(self):
        inout_pos = set(x.position for x in self.args if x.io == "input") & set(x.position for x in self.args if x.io == "output")
        self.inouts = {}
        for position in inout_pos:
            iarg = [x for x in self.args if x.position == position and x.io == "input"]
            oarg = [x for x in self.args if x.position == position and x.io == "output"]
            assert len(iarg) == 1 and len(oarg) == 1
            self.inouts[position] = (iarg[0], oarg[0])

    def emit(self, varset, cg):
        vargs = []
        inouts = set(x.position for x in self.args if x.io == "input") & set(x.position for x in self.args if x.io == "output")
        for a in it.ifilter(lambda x: x.position not in inouts or x.io == "output", self.args):
            if a.io == "input":
                vargs.append(varset.read_ref(a.arg, cg))
            elif a.io == "output":
                vargs.append(varset.write_ref(a.arg, cg, a.position != -1))
        if len(self.args) > 0 and self.args[0].position == -1:  # i.e. there is a return value we use
            return cg.assign(vargs[0], cg.invoke(self.rule.get_name(), vargs[1:]))
        else:
            return cg.invoke(self.rule.get_name(), vargs)

# TODO: I don't like this function being here, nor what it does, but it's a necessary evil while load_app/store_app call cgen directly.


def vectorize_symbolic_constant(t, vector_var):
    if isinstance(t, term.symbolic_constant) and t == vector_var:
        t = term.add([t, term.symbolic_constant("__hfav_vlane")])
    elif isinstance(t, term.add):
        t = term.add(t.args)
        t.args = [vectorize_symbolic_constant(x, vector_var) for x in t.args]
    elif isinstance(t, term.neg):
        t = term.neg(t.args)
        t.args = [vectorize_symbolic_constant(x, vector_var) for x in t.args]
    elif isinstance(t, term.at):
        t = term.at(t.args)
        t.args = [vectorize_symbolic_constant(x, vector_var) for x in t.args]
    return t


class load_app(rule_app_base):

    def __init__(self, ax, symb={}):
        super(load_app, self).__init__(ax, symb)
        self.types[self.antecedent[0]] = self.rule.type
        assert len(self.antecedent) == 1
        self.consequent = [x.substitute(self.sub) for x in self.rule.consequent]
        self.types[self.consequent[0]] = self.rule.type
        assert len(self.consequent) == 1

    def name(self):
        return "load"

    def split(self, split_vars):
        res = copy.deepcopy(self)
        return res

    def emit(self, varset, cg):
        actual_term = self.antecedent[0].args[0]
        if varset.is_input_variable(self.consequent[0]):
            return None
        if cg.vector_var is not None and cg.vectorize:
            rref = cg.read_aref(var_key_noarray(actual_term), reversed([vectorize_symbolic_constant(x, cg.vector_var) for x in actual_term.position_tuple()]))
        else:
            rref = cg.read_aref(var_key_noarray(actual_term), reversed(actual_term.position_tuple()))
        return cg.assign(varset.write_ref(self.consequent[0], cg), rref)


class enclosing_load_app(load_app):

    def __init__(self, lv, ttype):
        ax = axiom(term.function("_inref", [lv]), lv, ttype)
        super(enclosing_load_app, self).__init__(ax)

    def name(self):
        return "enclosing_load"

    def in_zone(self, varset, zone):
        """This is probably wrong; it assumes that the consqeuent will forward the storage."""
        return False


class ivar_app(load_app):

    def __init__(self, ax, symb={}):
        super(ivar_app, self).__init__(ax, symb)

    def name(self):
        return "ivar"

    def emit(self, varset, cg):
        return None


class reduction_initializer_app(rule_app):

    def __init__(self, rule, symb={}):
        super(reduction_initializer_app, self).__init__(rule, symb)

    def name(self):
        return "reduction_initializer"

    def emit(self, varset, cg):
        if cg.vector_var is not None and cg.vector_var in self.reduction_app.reduction_ivars():
            old_vectorize = cg.vectorize
            cg.vectorize = True
            res = "for (int __hfav_vlane = 0; __hfav_vlane < VLEN; ++__hfav_vlane) { %s; }" % cg.assign(varset.write_ref(self.consequent[0], cg), self.rule.op.identity)
            cg.vectorize = old_vectorize
            return res
        else:
            return cg.assign(varset.write_ref(self.consequent[0], cg), self.rule.op.identity)

    def in_zone(self, varset, zone):
        # Reduction initializations only appear in the prologue for the loop(s) they reduce over.
        zonedict = dict(zone)
        if any([riv not in zonedict for riv in self.reduction_app.reduction_ivars()]) or all([zonedict[riv] == 0 for riv in self.reduction_app.reduction_ivars()]):
            return super(reduction_initializer_app, self).in_zone(varset, zone)
        else:
            return False


class reduction_finalizer_app(rule_app):

    def __init__(self, rule, symb={}):
        super(reduction_finalizer_app, self).__init__(rule, symb)

    def name(self):
        return "reduction_finalizer"

    def emit(self, varset, cg):
        if cg.vector_var is not None and cg.vector_var in self.reduction_app.reduction_ivars():
            old_vectorize = cg.vectorize
            cg.vectorize = True
            lhs = varset.write_ref(self.consequent[0], cg)
            rhs = varset.read_ref(self.antecedent[0], cg)
            res = "for (int __hfav_vlane = 0; __hfav_vlane < VLEN; ++__hfav_vlane) { %s; }" % cg.assign(lhs, self.rule.op.fstring % (lhs, rhs))
            cg.vectorize = old_vectorize
            return res
        else:
            return cg.assign(varset.write_ref(self.consequent[0], cg), varset.read_ref(self.antecedent[0], cg))

    def in_zone(self, varset, zone):
        # Reduction finalizations only appear in the epilogue for the loop(s) they reduce over.
        zonedict = dict(zone)
        if any([riv not in zonedict for riv in self.reduction_app.reduction_ivars()]) or all([zonedict[riv] == 2 for riv in self.reduction_app.reduction_ivars()]):
            return super(reduction_finalizer_app, self).in_zone(varset, zone)
        else:
            return False


class store_app(rule_app_base):

    def __init__(self, goal, symb={}):
        super(store_app, self).__init__(goal, symb)
        self.types[self.antecedent[0]] = self.rule.type
        if len(self.antecedent) > 1:
            self.types[self.antecedent[1]] = self.rule.type
        assert len(self.antecedent) <= 2
        self.types[self.consequent[0]] = self.rule.type
        assert len(self.consequent) == 1

    def endpoint(self):
        return True

    def name(self):
        return "store"

    def split(self, split_vars):
        res = copy.deepcopy(self)
        sf = lambda x, y: term.function(x, [y]) if len(split_vars[var_key_noarray(y)]) > 0 else y
        res.types = {}
        res.antecedent = [sf("_post_split", ante) for ante in res.antecedent]
        res.types[res.antecedent[0]] = res.rule.type
        res.consequent = [sf("_pre_split", cons) for cons in res.consequent]
        res.types[res.consequent[0]] = res.rule.type
        return res

    def emit(self, varset, cg):
        actual_term = self.consequent[0].args[0]
        if varset.is_output_variable(self.antecedent[0]):
            return None
        if cg.vector_var is not None and cg.vectorize:
            wref = cg.write_aref(var_key_noarray(actual_term), reversed([vectorize_symbolic_constant(x, cg.vector_var) for x in actual_term.position_tuple()]))
        else:
            wref = cg.write_aref(var_key_noarray(actual_term), reversed(actual_term.position_tuple()))
        return cg.assign(wref, varset.read_ref(self.antecedent[0], cg))

    def in_zone(self, varset, zone):
        return not varset.is_output_variable(self.antecedent[0])


class enclosing_store_app(store_app):

    def __init__(self, st, ttype):
        ax = goal([st], [term.function("_outref", [st])], ttype)
        super(enclosing_store_app, self).__init__(ax)

    def name(self):
        return "enclosing_store"

    def in_zone(self, varset, zone):
        """This is probably wrong; it assumes that the antecedent will forward the storage."""
        return False


class codeblock_app(rule_app):

    def __init__(self, cb, symb={}):
        super(codeblock_app, self).__init__(cb, symb)

        self.outer_iv = cb.outer_iv
        self.inner_iv = cb.inner_iv

    def name(self):
        return "codeblock"

    def split(self, split_vars):
        res = copy.deepcopy(self)
        return res

    def emit(self, varset, cg):
        return self.rule.code

    def in_zone(self, varset, zone):
        return True

    def get_type(self, var):
        return self.types[var_key_noarray(var)]


class dag_chain(object):

    def __init__(self, pg, typedict, axioms):
        self.pg = pg
        self.typedict = typedict
        self.axioms = axioms

    def resolve(self, finals):
        dag = inference_dag(self.typedict)
        for next_consequent in finals:
            store = store_app(next_consequent, {term.variable("i?"): term.numeric_constant(0)})
            dag.rule_apply(store)
            dag.resolve_vertex(store.consequent[0], store.types[store.consequent[0]])
            dag = self.back_chain(dag, store.antecedent[0])
            if len(store.antecedent) > 1:
                try:
                    dag = self.back_chain(dag, store.antecedent[1])
                except chain_failed:
                    logger.debug("Got output variable.")
        return dag

    def back_chain(self, dag, desired_consequent):
        if isinstance(desired_consequent, term.numeric_constant):
            logger.debug("Found numeric constant")
            return dag
        if isinstance(desired_consequent, term.add) or isinstance(desired_consequent, term.neg):
            logger.debug("Found simple arithmetic expression")
            try:
                adag = dag.copy()
                for a in desired_consequent.args:
                    adag = self.back_chain(adag, a)
                return adag
            except chain_failed:
                logger.debug("Backtracking!")
                pass
        if dag.has_resolved_term(desired_consequent):
            logger.debug("Already in dag")
            return dag
        for ax in self.axioms:
            axsub = ax.is_in(desired_consequent)
            if axsub:
                # Check we haven't already generated the axiom edge (e.g. for an ivar_axiom).
                if edge(axsub.antecedent[0], axsub.consequent[0]).key() not in dag.edges:
                    logger.debug("Got axiom")
                    dag.rule_apply(axsub)
                else:
                    logger.debug("Already in dag")
                    dag.vertices[desired_consequent].resolved = True
                return dag
        unified = self.pg.unifications(desired_consequent)
        if len(unified) == 0:
            raise chain_failed
        for u in unified:
            try:
                udag = dag.copy()
                udag.rule_apply(u)
                logger.debug("Trying unification")
                for next_consequent in u.antecedent:
                    udag = self.back_chain(udag, next_consequent)
                return udag
            except chain_failed:
                logger.debug("Backtracking!")
                pass
        raise chain_failed


class rule_group(object):

    """ container for a bunch of rules to hand to chainer"""

    def __init__(self):
        self.rules = []

    def unifications(self, desired_result):
        """Find all rules that unify with desired_result"""
        # TODO: is this really necessary
        assert desired_result.groundedp()
        back = []
        for p in self.rules:
            prod = p.unify(desired_result)
            if prod is not None:
                back.append(rule_app.from_prod(p, prod))
        return back
