# hfav/ispace.py; iteration space manipulation tools

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

import logging
from . import term


class strided_interval(object):

    def __init__(self, start, end, stride=1):
        self.start = str(start)
        self.end = str(end)
        self.stride = stride

    def __str__(self):
        return "[%s:%s:%s]" % (self.start, self.end, self.stride)

    @classmethod
    def num(cls, n):
        return cls.__init__(n, n + 1)

    def sweep(self, n):
        return strided_interval(self.start + n, self.end + n, self.stride)

    def explicit(self):
        return range(self.start, self.end, self.stride)


class iteration_space(object):

    def __init__(self, loop_dict, loop_order):
        self.loop_dict = loop_dict
        self.loop_order = loop_order

    def dim(self):
        return len(self.loop_order)

    def strides(self):
        return [(k, self.loop_dict[k].stride) for k in self.loop_order]

    def copy(self):
        return iteration_space(self.loop_dict.copy(), list(self.loop_order))

    def map_offset(self, iter_var, offs, roll_var=None):
        if roll_var is not None and iter_var == roll_var:
            return offs
        else:
            loop = self.loop_dict[iter_var]
            if loop.stride != 1:
                res_str = "(%s-%s)/%s" % (iter_var, loop.start, loop.stride)
            else:
                res_str = "%s-%s" % (iter_var, loop.start)
            if offs != 0:
                return res_str + "+%d" % offs
            else:
                return res_str

    def interval(self, ivar):
        return self.loop_dict[ivar]

    def subspace(self, ivars):
        loop_dict = {}
        for iv in ivars:
            loop_dict[iv] = self.loop_dict[iv]
        loop_order = []
        for iv in self.loop_order:
            if iv in ivars:
                loop_order.append(iv)
        return iteration_space(loop_dict, loop_order)

    def is_iter(self, var):
        return term.symbolic_constant(var) in self.loop_dict

    @classmethod
    def from_yaml(cls, config):
        loop_stuff = config["codegen options"]["loops"]
        if not isinstance(loop_stuff, list):
            loop_stuff = [loop_stuff]

        loops = dict((term.symbolic_constant(x["iter_ident"]), strided_interval(x["start"], x["end"], x["stride"])) for x in loop_stuff)

        try:
            loop_order = list(reversed([term.symbolic_constant(x) for x in config["codegen options"]["loop order"]]))
        except KeyError:
            loop_order = list(reversed([x for x in loops]))

        logging.debug("loop_order: %s" % map(str, loop_order),)
        logging.debug("loops: %s" % (str(loops),))
        return cls(loops, loop_order)
