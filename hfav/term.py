# hfav/term.py; Basic types for inference

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

import operator as op
import logging
import copy
logger = logging.getLogger(__name__)


class term(object):

    def __init__(self):
        pass

    def name(self):
        return self.ident

    def __str__(self):
        return self.ident

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, term):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self == other

    def canonize(self):
        return self

    def position(self):
        return position(self.position_tuple())

    def position_tuple(self):
        return ()

    def occurp(self, o):
        return self == o

    def unify(self, y, symb={}, swap=False):
        if symb is None:
            return None
        elif self == y:
            return symb
        elif not swap:
            return y.unify(self, symb, True)
        return None

    def symbolic_constants(self):
        return []

    def hoist_positions(self):
        return copy.deepcopy(self)

    def bury_positions(self):
        return copy.deepcopy(self)


class atom(term):

    def __init__(self):
        pass

    def substitute(self, subs):
        return self

    def __repr__(self):
        return "%s(\"%s\")" % (self.__class__.__name__, self.ident)


class constant(atom):

    def __init__(self):
        pass

    def assoc_order(self):
        return 2

    def groundedp(self):
        return True


class symbolic_constant(constant):

    def __init__(self, ident):
        assert isinstance(ident, str)
        self.ident = ident

    def __lt__(self, other):
        if not isinstance(self, add) and isinstance(other, add):
            return add.identity([self]) < other
        elif isinstance(other, symbolic_constant):
            return self.ident < other.ident
        elif isinstance(other, numeric_constant):
            return False
        elif isinstance(other, term):
            return True
        else:
            raise TypeError("Tried to compare a %s to a %s" % (type(self), type(other)))

    def symbolic_constants(self):
        return [self]


class numeric_constant(constant):

    def __init__(self, value):
        self.value = value

    def name(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return "%s(%d)" % ("numeric_constant", self.value)

    def __lt__(self, other):
        if not isinstance(self, add) and isinstance(other, add):
            return add.identity([self]) < other
        elif isinstance(other, numeric_constant):
            return self.value < other.value
        elif isinstance(other, term):
            return True
        else:
            raise TypeError("Tried to compare a %s to a %s" % (type(self), type(other)))


class variable(atom):

    def __init__(self, ident):
        self.ident = ident

    def __lt__(self, other):
        if isinstance(other, variable):
            return self.ident < other.ident
        elif isinstance(other, function):
            return True
        elif isinstance(other, term):
            return False
        else:
            raise TypeError("Tried to compare a %s to a %s" % (type(self), type(other)))

    def assoc_order(self):
        return 1

    def groundedp(self):
        return False

    def substitute(self, subs):
        if self in subs:
            return subs[self]
        return self

    def unify(self, x, symb={}, swap=False):
        su = super(variable, self).unify(x, symb, False)
        if su:
            return su
        elif self in symb:
            return symb[self].unify(x, symb, False)
        elif x in symb:
            return self.unify(symb[name(x)], symb, False)
        elif self.occurp(x):
            return None
        else:
            z = symb.copy()
            z[self] = x
            return z

    def as_symbolic_constant(self):
        return symbolic_constant(self.ident.strip("?"))


class function(term):

    def __init__(self, ident, args):
        self.ident = ident
        assert ident != "at"
        self.args = list(args)

    def __str__(self):
        return self.ident + "(" + ", ".join((str(x) for x in self.args)) + ")"

    def __repr__(self):
        return "%s(\"%s\", %r)" % ("function", self.ident, self.args)

    def __lt__(self, other):
        if not isinstance(self, add) and isinstance(other, add):
            return add.identity([self]) < other
        elif isinstance(other, function):
            if not self.ident == other.ident:
                return self.ident < other.ident
            else:
                lmax = max(len(self.args), len(other.args))
                sargs = self.args + [numeric_constant(0)] * (lmax - len(self.args))
                oargs = other.args + [numeric_constant(0)] * (lmax - len(other.args))
                for n in range(lmax):
                    if not sargs[n] == oargs[n]:
                        return sargs[n] < oargs[n]
            return False
        elif isinstance(other, term):
            return False
        else:
            raise TypeError("Tried to compare a %s to a %s" % (type(self), type(other)))

    def position_tuple(self):
        if len(self.args) > 1:
            pt = None
            for a in self.args:
                this_pt = a.position_tuple()
                if pt is None:
                    pt = this_pt
                if this_pt and this_pt != pt:
                    raise ValueError("Mismatched dig positions in multi-argument function!")
            return pt
        elif len(self.args) == 0:
            raise ValueError("Can't dig position out of 0-argument function")
        else:
            return self.args[0].position_tuple()

    def assoc_order(self):
        return 0

    def groundedp(self):
        return all((x.groundedp() for x in self.args))

    def canonize(self):
        return function(self.ident, (x.canonize() for x in self.args))

    def occurp(self, o):
        return any((x.occurp(o) for x in self.args))

    def substitute(self, subs):
        f = function(self.ident, (x.substitute(subs) for x in self.args))
        return f.canonize()

    def unify(self, x, symb={}, swap=False):
        su = super(function, self).unify(x, symb, swap)
        if su:
            return su
        elif isinstance(x, function) and self.ident == x.ident and len(self.args) == len(x.args):
            for ma, oa in zip(self.args, x.args):
                symb = ma.unify(oa, symb, False)
                if symb is None:
                    return None
            return symb
        return None

    def symbolic_constants(self):
        l = []
        for a in self.args:
            l += a.symbolic_constants()
        return l

    def hoist_positions(self):
        ret = copy.deepcopy(self)
        if len(ret.args) == 1:
            if not isinstance(ret.args[0], at):
                ret.args[0] = ret.args[0].hoist_positions()
            if isinstance(ret.args[0], at):
                outer = ret.args[0]
                ret.args[0] = outer.args[0]
                outer.args[0] = ret.hoist_positions()
                return outer
        return ret

    def bury_positions(self):
        ret = copy.deepcopy(self)
        if len(ret.args) == 1:
            ret.args[0] = ret.args[0].bury_positions()
        return ret


def add_unify_helper(t):
    # this has the known limitation that it will not unify functions, etc below
    # addition. Sorry, too hard!
    mvar = None
    mterms = []
    assert isinstance(t, add)
    for i in t.args:
        if isinstance(i, function) and not i.groundedp():
            return None
        elif isinstance(i, variable):
            if mvar is None:
                mvar = i
                continue
            else:
                return None
        mterms.append(i)
    return mvar, mterms


class add(function):

    def __init__(self, args):
        self.ident = "add"
        self.args = args

    @classmethod
    def identity(cls, args):
        assert not isinstance(args[0], cls)
        return add(args + [numeric_constant(0)])

    def canonize(self):
        tp = []
        sum = 0
        for i in self.args:
            ic = i.canonize()
            if ic is None:
                continue
            if isinstance(ic, add):
                tp += ic.args
            elif isinstance(ic, numeric_constant):
                sum += ic.value
            else:
                tp.append(i)
        tp0 = sorted(tp, key=op.attrgetter('name'))
        tp = sorted(tp0, key=op.attrgetter('assoc_order'))
        if sum != 0:
            tp.append(numeric_constant(sum))
        if len(tp) == 0:
            return numeric_constant(0)
        elif len(tp) == 1:
            return tp[0]
        else:
            return add(tp)

    def __lt__(self, other):
        if not isinstance(other, add):
            other = add.identity([other])
        return super(add, self).__lt__(other)

    def __str__(self):
        return "+".join((str(x) for x in self.args)).replace("+-", "-")

    def __repr__(self):
        return "%s(%r)" % (self.ident, self.args)

    def substitute(self, subs):
        f = add((x.substitute(subs) for x in self.args))
        return f.canonize()

    def unify(self, other, symb={}, swap=False):
        # try to solve for a variable in one or the other
        mcpy = self.substitute(symb)
        mt = add_unify_helper(mcpy)
        if not mt:
            return None
        ocpy = other.substitute(symb)
        if not isinstance(ocpy, add):
            ocpy = add([ocpy, numeric_constant(0)])
        ot = add_unify_helper(ocpy)
        if not ot:
            return None
        (mtv, mtr) = mt
        (otv, otr) = ot
        z = symb.copy()
        if mtv:
            nf = [neg([x]) for x in mtr] + otr
            if otv:
                nf.append(otv)
            z[mtv] = add(nf).canonize()
        elif otv:
            nf = [neg([x]) for x in otr] + mtr
            if mtv:
                nf.append(mtv)
            z[otv] = add(nf).canonize()
        elif mtr == otr:
            return symb
        else:
            return None
        return z


class neg(function):

    def __init__(self, args):
        self.ident = "neg"
        self.args = args

    def canonize(self):
        children = [x.canonize() for x in self.args]
        if len(children) == 0:
            return numeric_constant(0)
        elif len(children) == 1:
            if isinstance(children[0], numeric_constant):
                return numeric_constant(-children[0].value)
            elif isinstance(children[0], neg):
                return numeric_constant(children[0])
            else:
                return neg(children)
        else:
            children = [children[0]] + [neg([x]).canonize() for x in children[1:]]
            return add(children).canonize()

    def __str__(self):
        if len(self.args) > 1:
            return "-".join((str(x) for x in self.args))
        else:
            return "-" + str(self.args[0])

    def __repr__(self):
        return "%s(%r)" % (self.ident, self.args)

    def substitute(self, subs):
        f = neg((x.substitute(subs) for x in self.args))
        return f.canonize()


class at(function):

    def __init__(self, var, idx):
        self.ident = "at"
        self.args = [var, idx]

    def bury_positions(self):
        self.args[0] = self.args[0].bury_positions()
        if isinstance(self.args[0], function) and not isinstance(self.args[0], at):
            outer = self.args[0]
            self.args[0] = outer.args[0]
            outer.args[0] = self.bury_positions()
            return outer
        return self

    def canonize(self):
        """TODO: Is there a way I can have python handle the class stuff here properly?"""
        return at(*[x.canonize() for x in self.args])

    def substitute(self, subs):
        f = at(*list((x.substitute(subs) for x in self.args)))
        return f.canonize()

    def position_tuple(self):
        ### This is inefficient
        return tuple([self.args[1]] + list(self.args[0].position_tuple()))

    def __str__(self):
        return str(self.args[0]) + "[%s]" % (str(self.args[1]),)

    def __repr__(self):
        return "%s(%r, %r)" % (self.ident, self.args[0], self.args[1])


class position(object):

    def __init__(self, pos):
        self.items = [simple_offset(p) for p in pos]
        self.named = {}
        for c, (v, o) in enumerate(self.items):
            if isinstance(v, symbolic_constant):
                assert not v in self.named
                self.named[v] = c

    def __eq__(self, o):
        if isinstance(o, position):
            return self.items == o.items and self.named == o.named
        else:
            print "tried to compare %s to %s (%s to %s)" % (self, o, type(self), type(o))
            return False

    def __hash__(self):
        return hash(tuple(self.items))

    def __str__(self):
        return "[" + ", ".join(("(%s: %s)" % (str(var), str(offset)) for var, offset in self.items)) + "]"

    def empty(self):
        return len(self.items) == 0

    def var_match(self, opos):
        return self.named == opos.named

    def var_order(self):
        if self.empty():
            return []
        else:
            return zip(*sorted(self.named.items(), key=op.itemgetter(1)))[0]

    def iter_perm(self, iter_order):
        return iteration_permutation(iter_order, self.var_order())

    def plotting_point(self):
        return (self.items[self.named[symbolic_constant("i")]][1], self.items[self.named[symbolic_constant("j")]][1])

    def variables(self):
        return set(self.named.keys())

    def ordered_variables(self):
        return sorted(self.named.items(), key=op.itemgetter(1))

    def difference(self, opos, strides=None):
        assert self.var_match(opos)
        if strides is not None:
            perm, active = self.iter_perm(s[0] for s in strides)
            active_strides = [strides[i] for i in active]
        else:
            perm = range(len(self.items))
            active_strides = strides
        res = []
        for i in range(len(self.items)):
            si = self.items[perm[i]]
            oi = opos.items[perm[i]]
            if strides is None:
                di = (si[0], 1)
            else:
                di = active_strides[i]
            assert si[0] == oi[0]
            assert si[0] == di[0]
            res.append((si[0], (si[1] - oi[1]) / float(di[1])))
        return tuple(res)


def simple_offset(pe):
    if isinstance(pe, symbolic_constant):
        return (pe, 0)
    elif isinstance(pe, numeric_constant):
        return (None, pe.value)
    elif isinstance(pe, add) and isinstance(pe.args[0], symbolic_constant) and isinstance(pe.args[1], numeric_constant):
        return (pe.args[0], pe.args[1].value)
    raise ValueError("not a simple offset")


def iteration_permutation(iter_order, var_order):
    res = []
    vars_used = []
    for p, symb in enumerate(iter_order):
        try:
            res.append(var_order.index(symb))
            vars_used.append(p)
        except ValueError:
            pass
    return res, vars_used
