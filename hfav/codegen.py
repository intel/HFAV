# hfav/codegen.py; code generation base class

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

from operator import attrgetter
import os
import logging
logger = logging.getLogger(__name__)


class codegen(object):

    def __init__(self, root, storage):
        self.root = root
        self.typedict = {'char': ['int', 8], 'byte': ['int', 8], 'int': ['int', 32], 'long': ['int', 64], 'float': ['float', 32], 'double': ['float', 64]}
        self.vector_var = None
        self.vectorize = False
        self.hindent = 0
        self.debug_vector_var = None  # TODO: Sorry Jason
        self.storage = storage
        pass

    def byref(self, ident):
        return "&" + ident

    def header(self, lst, h):
        self.rotate_function(lst)
        if h is not None:
            lst.append(h)
            opened = h.count('{')
            closed = h.count('}')
            self.hindent = opened - closed
            for i in range(0, opened - closed):
                lst.indent()

    def footer(self, lst, f):
        if f is not None:
            if self.hindent > 0:
                lst.deindent()
            lst.append(f)

    def ident_offset(self, ident, offset):
        if offset > 0:
            return ident + "+" + str(offset)
        elif offset == 0:
            return ident
        else:
            return ident + "-" + str(abs(offset))

    def prologue_gen(self):
        return self

    def epilogue_gen(self):
        return None

    def read_aref(self, ident, offset):
        return "%s%s" % (ident, "".join([("[%s]" % o) for o in offset]))

    def write_aref(self, ident, offset):
        return "%s%s" % (ident, "".join([("[%s]" % o) for o in offset]))

    def read_ref(self, ident):
        return "%s" % (ident,)

    def write_ref(self, ident):
        return "%s" % (ident,)

    def assign(self, dst, src):
        return "%s = %s" % (dst, src)

    def invoke(self, ident, args):
        return "%s(%s)" % (ident, ", ".join(args))

    def array_declaration(self, type, ident, size):
        if self.storage == "stack":
            return "%s %s%s" % (type, ident, "".join([("[%s]" % s) for s in size]))
        else:

            if size == []:
                return "%s %s" % (type, ident)

            if len(size) > 1:
                unroll_str = "".join([("[%s]" % s) for s in size[1:]])
            else:
                unroll_str = ""
            decl = "%s (*%s)%s" % (type, ident, unroll_str)
            cast = "%s(*)%s" % (type, unroll_str)
            flatsize = "*".join([("(%s)" % s) for s in size])
            return "%s = (%s) _mm_malloc((%s)*sizeof(%s), 64)" % (decl, cast, flatsize, type)

    def array_free(self, type, ident, size):
        if self.storage == "stack":
            return None
        else:
            if size == []:
                return None
            else:
                return "_mm_free(%s)" % (ident)

    def array_ptr_declaration(self, type, ptr_ident, src_ident, size):
        roll_str = "[%s]" % str(size[0])
        if len(size) > 2:
            unroll_str = "".join([("[%s]" % s) for s in size[2:]])
        else:
            unroll_str = ""
        dst = "%s (*%s%s)%s " % (type, ptr_ident, roll_str, unroll_str)
        srcs = []
        for r in range(size[0]):
            srcs.append(src_ident + ("[%s]" % str(r)))
        return self.assign(dst, "{" + ", ".join(srcs) + "}")

    def statement(self, lst, state):
        if state is not None:
            lst.append(state + ";\n")

    def init_iters(self, lst, loops):
        if len(loops.loop_dict.keys()) > 0:
            lst.append("int %s;\n" % (", ".join(map(attrgetter("ident"), loops.loop_dict.keys()))))

    def begin_scope(self, lst):
        lst.append("{\n")
        lst.indent()

    def end_scope(self, lst):
        lst.deindent()
        lst.append("}\n")

    def begin_loop(self, lst, itervar, interval, phase):
        stride = interval.stride
        if phase == [0]:
            lst.append("%s = %s;\n" % (itervar, interval.start))
            lst.append("if (%s < %s)\n" % (itervar, interval.end))
        elif 1 in phase:
            if self.debug_vector_var is not None and self.debug_vector_var == itervar:
                lst.append("#pragma simd assert\n")
            start = interval.start if 0 in phase else "%s+%s" % (interval.start, stride)
            end = interval.end if 2 in phase else "%s-%s" % (interval.end, stride)
            lst.append("for (%s = %s; %s < %s; %s += %s)\n" % (itervar, start, itervar, end, itervar, stride))
        elif phase == [2]:
            lst.append("%s = %s-1;\n" % (itervar, interval.end))
            lst.append("if (%s > %s)\n" % (itervar, interval.start))
        self.begin_scope(lst)

    def end_loop(self, lst, itervar, interval, phase):
        self.end_scope(lst)

    def rotate_header(self):
        return "\n"  # TODO: Decide if we should remove this completely.
        # return "#include \"hfav/c99-rotate.h\"\n"

    def rotate_function(self, lst):
        lst.append(self.rotate_header())

    def rotate(self, type, ident, start, end, roll_var):
        return self.invoke("rotate_%s%s" % (self.typedict[type][0], self.typedict[type][1]), [ident, str(start), str(end), "1"])

    def rotate_ptr(self, type, ident, len, roll_var):
        return self.invoke("rotate_%s%s_ptr" % (self.typedict[type][0], self.typedict[type][1]), [ident, str(len)])

    def comment(self, lst, lines):
        if len(lines) > 2:
            lst.append("/* " + lines[0] + "\n")
            for li in ["   " + z + "\n" for z in lines[1:-1]]:
                lst.append(li)
            lst.append("   " + lines[-1] + "*/\n")
        else:
            for li in ["// " + z + "\n" for z in lines]:
                lst.append(li)


class listing(object):

    def __init__(self):
        self.indent_level = 0
        self.lines = []
        self.indent_width = 4

    def indent(self):
        self.indent_level += 1

    def deindent(self):
        self.indent_level -= 1

    def append(self, string):
        assert string[-1] == '\n'
        self.lines.append(self.indent_level * self.indent_width * " " + string)

    def emit(self):
        return "".join(self.lines)
