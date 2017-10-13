# hfav/parse.py; Parse iteration/variable descriptions

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

from . import term


class parse_error(ValueError):
    pass

class fatal_parse_error(ValueError):
    pass

"""grammar:
<numeric-literal> := <digit>+
<symbolic-const>  := <alpha>[<alpha>|<digit>]*['!']
<const>           := [<symbolic-const>|<numeric-literal>]
<variable>        := <alpha>[<alpha>|<digit>]*'?'
<function-identifier> := <alpha>[<alpha>|<digit>]*'
<function>        := <function-identifier>'('<expr-list>')'
<term>            := [<function>|<const>|<variable>]
<prefix-op>       := ['-'|'+']
<infix-op>        := ['-'|'+']
<suffix-op>       := '['<expr>']'
(had to modify to remove a left-recursive term, see below)
<expr>            := [<term>|<prefix-op><expr>|<expr><infix-op><expr>|<expr><suffix-op>]
<expr-list>       := [<expr>?|','<expr-list>]

this
<expr>            := [<term>|<prefix-op><expr>|<expr><infix-op><expr>|<expr><suffix-op>]

is left-recusive, so we modify to have:
<expr>            := [<term><tail-expr>|<prefix-op><expr><tail-expr>]
<tail-expr>       := [<>|<infix-op><tail-expr>|<suffix-op><tail-expr>]
"""


class parser(object):

    """Each parser either consumes some of string (advances pos) and returns something or resets pos and throws a parser_error"""

    def __init__(self, string):
        self.string = string
        self.pos = 0

    def digit(self):
        c = self.string[self.pos:self.pos + 1]
        if c.isdigit():
            self.pos += 1
            return c
        raise parse_error("Not a digit")

    def alpha(self):
        c = self.string[self.pos:self.pos + 1]
        if c.isalpha() or c == '_':
            self.pos += 1
            return c
        raise parse_error("Not an alpha character")

    def alpha_digit(self):
        c = self.string[self.pos:self.pos + 1]
        if c.isalpha() or c == '_' or c.isdigit():
            self.pos += 1
            return c
        raise parse_error("Not an alpha or digit character")

    def match(self, char):
        c = self.string[self.pos:self.pos + 1]
        if c == char:
            self.pos += 1
            return c
        raise parse_error("Not a %s" % (char,))

    def whitespace(self):
        while self.pos < len(self.string) and self.string[self.pos].isspace():
            self.pos += 1

    def numeric_literal(self):
        """<numeric-literal> := <digit>+"""
        pos = self.pos
        res = []
        try:
            while True:
                res.append(self.digit())
        except parse_error:
            if len(res) < 1:
                self.pos = pos
                raise parse_error("Not a number")
            return term.numeric_constant(int(str(''.join(res)))).canonize()

    def symbolic_const(self):
        """<symbolic-const> := <alpha>[<alpha>|<digit>]*['!']"""
        pos = self.pos
        res = []
        res.append(self.alpha())
        try:
            while True:
                res.append(self.alpha_digit())
        except parse_error:
            try:
                res.append(self.match('!'))
            except parse_error:
                pass
            return term.symbolic_constant(''.join(res)).canonize()

    def const(self):
        """<const> := [<symbolic-const>|<numeric-literal>]"""
        pos = self.pos
        try:
            try:
                return self.symbolic_const()
            except parse_error:
                self.pos = pos
                return self.numeric_literal()
        except parse_error:
            self.pos = pos
            raise parse_error("Not a const")

    def variable(self):
        """<variable> := <alpha>[<alpha>|<digit>]*'?'"""
        pos = self.pos
        try:
            res = []
            res.append(self.alpha())
            try:
                while True:
                    res.append(self.alpha_digit())
            except parse_error:
                res.append(self.match('?'))
                return term.variable(''.join(res)).canonize()
        except parse_error:
            self.pos = pos
            raise parse_error("Not a variable")

    def function_identifier(self):
        """<function-identifier> := <alpha>[<alpha>|<digit>]*"""
        pos = self.pos
        try:
            res = []
            res.append(self.alpha())
            try:
                while True:
                    res.append(self.alpha_digit())
            except parse_error:
                return ''.join(res)
        except parse_error:
            self.pos = pos
            raise parse_error("Not a function identifier")

    def function(self):
        """<function> := <function-identifier>'('<expr-list>')'"""
        pos = self.pos
        try:
            fi = self.function_identifier()
            self.whitespace()
            self.match('(')
            try:
                li = self.expr_list()
                self.whitespace()
                self.match(')')
            except parse_error:
                raise fatal_parse_error("Not a valid expression list.")
            return term.function(fi, li).canonize()
        except parse_error:
            self.pos = pos
            raise parse_error("Not a function")

    def expr_list(self):
        """<expr-list> := [<expr>?|','<expr-list>]"""
        pos = self.pos
        res = []
        try:
            res.append(self.expr())
            while True:
                self.whitespace()
                self.match(',')
                res.append(self.expr())
        except parse_error:
            return res

    def term(self):
        """<term> := [<function>|<const>|<variable>]"""
        pos = self.pos
        self.whitespace()
        try:
            return self.function()
        except parse_error:
            pass
        try:
            return self.variable()
        except parse_error:
            pass
        try:
            return self.const()
        except parse_error:
            pass
        self.whitespace()

        self.pos = pos
        raise parse_error("Not a term")

    def prefix_op(self):
        """<prefix-op> := ['-'|'+']"""
        try:
            self.match('-')
            return term.neg
        except parse_error:
            try:
                self.match('+')
                return term.add
            except parse_error:
                raise parse_error("not a prefix op")

    def infix_op(self):
        """<infix-op> := ['-'|'+']"""
        try:
            self.match('+')
            return term.add
        except parse_error:
            pass
        try:
            self.match('-')
            return term.neg
        except parse_error:
            raise parse_error("not an infix op")

    def suffix_op(self, left):
        """<suffix-op> := '['<expr>']'"""
        pos = self.pos
        try:
            self.whitespace()
            self.match('[')
            expr1 = self.expr()
            self.match(']')
            return term.at(left, expr1).canonize()
        except parse_error:
            self.pos = pos
            raise parse_error("Not a suffix op")

    def tail_expr(self, left):
        """<tail-expr> := [<EOF>|<infix-op><expr><tail-expr>|<suffix-op><tail-expr>]"""
        pos = self.pos
        try:
            self.whitespace()
            op = self.infix_op()
            expr0 = self.expr()
            newleft = op([left, expr0])
            return self.tail_expr(newleft)
        except parse_error:
            pass
        try:
            self.whitespace()
            op = self.suffix_op(left)
            return self.tail_expr(op)
        except parse_error:
            self.pos = pos
            return left

    def expr(self):
        """<expr> := <term><tail-expr>|<prefix-op><expr><tail-expr>"""
        pos = self.pos
        try:
            self.whitespace()
            a = self.term()
            self.whitespace()
            return self.tail_expr(a)
        except parse_error:
            pos = self.pos
        try:
            self.whitespace()
            a = self.prefix_op()
            b = self.expr()
            return a([self.tail_expr(b)]).canonize()
        except parse_error:
            pos = self.pos
        raise parse_error("no expression found")

if __name__ == '__main__':
    print(parser(" a (x? , a(q![i!-1] ) [ 1 + 1 ], -1 )").expr())
