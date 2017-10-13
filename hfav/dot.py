# hfav/dot.py; graphviz 'dot' output

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

from . import codegen


class dot_generator(codegen.codegen):

    def __init__(self):
        self.lines = []
        self.indent = 0

    def header(self, lst):
        lst.append("digraph\n")
        lst.append("{\n")
        lst.indent()
        lst.append("size=\"20,20\";\n")
        lst.append("ratio=fill;\n")
        lst.append("node [shape=box];\n")

    def footer(self, lst):
        lst.append("}\n")
        lst.deindent()

    def offset_string(self, offset):
        offstring = str(offset)
        offstring = offstring.replace("+", "p")
        return offstring.replace("-", "m")

    def read_aref(self, ident, offset):
        return "%s_%s" % (ident, self.offset_string(offset))

    def write_aref(self, ident, offset):
        return "%s_%s" % (ident, self.offset_string(offset))

    def assign(self, dst, src):
        return "%s -> %s" % (src, dst)

    def invoke(self, ident, outputs, inputs):
        assignments = []
        first = 1
        for i in inputs:
            for o in outputs:
                if first == 1:
                    assignments.append(self.assign(o, i))
                    first = 0
                else:
                    assignments.append(self.assign(o, i))
        return "\n".join(assignments)[:-1]  # hackily remove trailing newline

    def array_declaration(self, ident, size):
        declarations = []
        for offset in range(0, size):
            declarations.append(self.indent * " " + "%s_%s [label=\"%s[%s]\"];" % (ident, self.offset_string(offset), ident, offset))
        return "\n".join(declarations)[:-1]

    def begin_loop(self, lst, loopi):
        pass

    def end_loop(self, lst):
        pass
