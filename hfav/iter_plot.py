# hfav/iter_plot.py; iteration space plotting tools

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

import matplotlib
matplotlib.use('agg')
import pylab
import math
import itertools as it
import logging

import matplotlib.patches as mpatches

radius = 0.1
hl = 0.1


def arrow(s, e, estyle):
    delta = tuple((ev - sv for sv, ev in zip(s, e)))
    l = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1])
    unit = (delta[0] / l, delta[1] / l)
    start = (s[0] + unit[0] * radius, s[1] + unit[1] * radius)
    edel = (delta[0] - unit[0] * (hl + 2 * radius), delta[1] - unit[1] * (hl + 2 * radius))
    return mpatches.FancyArrow(start[0], start[1], edel[0], edel[1], head_length=hl, head_width=0.05, **estyle)


def iter_plot_start():
    fig = pylab.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    return ax


def iter_plot(ax, dag, vstyle={}, estyle={}):
    seen = set()

    patches = []
    for s, e in dag.edges.keys():
        ps = s.plotting_point()
        pe = e.plotting_point()
        seen.add(ps)
        seen.add(pe)
        ar = arrow(ps, pe, estyle)
        ax.add_patch(ar)

    xmin = None
    xmax = None
    ymin = None
    ymax = None
    for p in (x.plotting_point() for x in dag.vertices.keys()):
        if xmin is None or p[0] < xmin:
            xmin = p[0]
        if xmax is None or p[0] > xmax:
            xmax = p[0]

        if ymin is None or p[1] < ymin:
            ymin = p[1]
        if ymax is None or p[1] > ymax:
            ymax = p[1]
        point = mpatches.Circle(p, 0.1, **vstyle)
        ax.add_patch(point)

    xticks = range(xmin - 1, xmax + 2)
    yticks = range(ymin - 1, ymax + 2)

    pylab.xticks(xticks, xticks)
    pylab.yticks(yticks, yticks)

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    ax.set_aspect('equal')


def iter_plot_finish(fp):
    pylab.savefig(fp)
