# hfav/old_cpp.py; Unmaintained C++ code generation

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

from . import c99
import os


class cpp_generator(c99.c99_generator):

    def __init__(self, root):
        super(cpp_generator, self).__init__(root)
        pass

    def byref(self, ident):
        return ident

    def rotate_header(self):
        return "\n"
        #return "#include \"hfav/cpp-rotate.hpp\"\n"

    def rotate(self, type, ident, start, end, roll_var):
        return self.invoke("rotate", [ident, str(start), str(end), "1"])

    def rotate_ptr(self, type, ident, len, roll_var):
        return self.invoke("rotate_ptr", [ident, str(len)])


class cpp_autovec_generator(cpp_generator):

    def __init__(self, root, vector_var):
        super(cpp_autovec_generator, self).__init__(root, vector_var)
        pass

    def rotate(self, type, ident, start, end, roll_var):
        if roll_var == self.vector_var:
            if self.vectorize:
                return self.invoke("rotate", [ident, str(start), str(end), "VLEN"])
            else:
                return super(cpp_autovec_generator, self).rotate(type, ident, start, end, roll_var)
        elif self.vector_var is not None:
            return self.invoke("vrotate", [ident, str(start), str(end), "1"])
        else:
            return super(cpp_autovec_generator, self).rotate(type, ident, start, end, roll_var)

    def rotate_ptr(self, type, ident, len, roll_var):
        if roll_var == self.vector_var:
            if self.vectorize:
                raise NotImplementedError("rotate_ptr is not implemented for vector types")
            else:
                return super(cpp_autovec_generator, self).rotate_ptr(type, ident, len, roll_var)
        else:
            return super(cpp_autovec_generator, self).rotate_ptr(type, ident, len, roll_var)
