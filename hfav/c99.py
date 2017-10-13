# hfav/c99.py; code generation for c99

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
import os


class c99_generator(codegen.codegen):

    def __init__(self, root, storage, vector_var):
        super(c99_generator, self).__init__(root, storage)
        self.vector_var = vector_var
        self.remainder = False
        pass

    def begin_vector_loop(self, lst):
        lst.append("#pragma simd assert\n")
        lst.append("for (int __hfav_vlane = 0; __hfav_vlane < VLEN; ++__hfav_vlane)\n")
        self.begin_scope(lst)

    def end_vector_loop(self, lst):
        self.end_scope(lst)

    def begin_loop(self, lst, itervar, interval, phase):
        if not (itervar == self.vector_var and 1 in phase):
            super(c99_generator, self).begin_loop(lst, itervar, interval, phase)
        else:
            stride = str(interval.stride)
            vstride = stride + "*VLEN"
            start = interval.start if 0 in phase else "%s+%s" % (interval.start, stride)
            end = interval.end if 2 in phase else "%s-%s" % (interval.end, stride)
            vbound = "%s + (((%s)-(%s)) & ~(VLEN-1))" % (start, end, start)
            lst.append("const int %s_vbound = %s;\n" % (itervar, vbound))
            lst.append("for (%s = %s; %s < %s_vbound; %s += %s)\n" % (itervar, start, itervar, itervar, itervar, vstride))
            self.vectorize = True
            self.begin_scope(lst)

    def end_loop(self, lst, itervar, interval, phase):
        if (itervar == self.vector_var and 1 in phase):
            self.vectorize = False
        super(c99_generator, self).end_loop(lst, itervar, interval, phase)

    def begin_remainder_loop(self, lst, itervar, interval, phase):
        stride = interval.stride
        start = interval.start if 0 in phase else "%s+%s" % (interval.start, stride)
        end = interval.end if 2 in phase else "%s-%s" % (interval.end, stride)
        vbound = vbound = "%s + (((%s)-(%s)) & ~(VLEN-1))" % (start, end, start)
        lst.append("for (%s = %s_vbound; %s < %s; %s += %s)\n" % (itervar, itervar, itervar, end, itervar, interval.stride))
        self.begin_scope(lst)
        self.remainder = True

    def end_remainder_loop(self, lst, itervar, interval, phase):
        self.end_scope(lst)
        self.remainder = False

    def rotate(self, type, ident, start, end, roll_var):
        if roll_var == self.vector_var:
            if self.vectorize:
                return self.invoke("rotate_%s%s" % (self.typedict[type][0], self.typedict[type][1]), [ident, str(start), str(end), "VLEN"])
            else:
                return super(c99_generator, self).rotate(type, ident, start, end, roll_var)
        elif self.vector_var is not None:
            return self.invoke("vrotate_%s%s" % (self.typedict[type][0], self.typedict[type][1]), [ident, str(start), str(end), "1"])
        else:
            return super(c99_generator, self).rotate(type, ident, start, end, roll_var)

    def rotate_ptr(self, type, ident, len, roll_var):
        if roll_var == self.vector_var:
            if self.vectorize:
                raise NotImplementedError("rotate_ptr is not implemented for vector types")
            else:
                return super(c99_generator, self).rotate_ptr(type, ident, str(len), roll_var)
        else:
            return super(c99_generator, self).rotate_ptr(type, ident, str(len), roll_var)
