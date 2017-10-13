// examples/laplace5-test/laplace5-test.cpp; 5-point laplace stencil codegen example

// Copyright 2017 Intel Corporation
//
// GENERATED CODE EXEMPTION
//
// The output of this tool does not automatically import the Apache
// 2.0 license, except the output will continue to be subject to the
// limitation of liability clause in the Apache 2.0 license. Users may
// license their output under any license they choose but the liability
// of the authors of the tool for that output is governed by the
// limitation of liability clause in the Apache 2.0 license.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>

static int GD;

static double h;
static double h_inv;

static void laplace5_resid(double rhs, double n, double ne, double e, double se, double s, double sw, double w, double nw, double self, double* out)
{
    *out = rhs - h_inv*h_inv*(n + e + s + w - 4*self);
}

static double L2norm(const int GD, const double in[restrict][GD+2])
{
    double res = 0.0;
    for(int j = 1; j < GD+1; ++j)
    {
        for(int i = 1; i < GD+1; ++i)
        {
            double nv;
            laplace5_resid(0.0, in[j-1][i], in[j-1][i+1], in[j][i+1], in[j+1][i+1], in[j+1][i], in[j+1][i-1], in[j][i-1], in[j-1][i-1], in[j][i], &nv);

            res += nv*nv;
        }
    }
    return std::sqrt(res);
}

static double omega = 2.0/3.0;

static void laplace5(double n, double e, double s, double w, double self, double* out)
{
    *out =  (1.0 - omega) * self + omega*h*h/4.0*(0.0 - h_inv*h_inv*(n + e + s + w));
}

#ifdef USE_GEN
#define VLEN 4
#include "hfav/c99-rotate.h"
#include "laplace5-gen.hpp"
#endif

static void test_compute(const int GD, const double in[restrict][GD+2], double out[restrict][GD+2])
{
    for(int j = 1; j < GD+1; ++j)
    {
        for(int i = 1; i < GD+1; ++i)
        {
            laplace5(in[j-1][i], in[j][i+1], in[j+1][i], in[j][i-1], in[j][i], &out[j][i]);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s [# iterations] [size]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const int iterations = atoll(argv[1]);
    GD                   = atoll(argv[2]);

    h = 1.0/GD;
    h_inv = 1.0/h;

    // pad by 1 on each side so we don't have to branch in operatorr
    double *in  = (double*) calloc((GD+2) * (GD+2), sizeof(double));
    double *out = (double*) calloc((GD+2) * (GD+2), sizeof(double));

    typedef double grid[GD+2][GD+2];

    srand(12345);
    for(int j = 1; j < GD+1; ++j)
    {
        for(int i = 1; i < GD+1; ++i)
        {
            in[j*(GD+2) + i] = drand48()*h_inv*h_inv;
        }
    }

    for(int j = 0; j < GD+2; ++j)
    {
        in[j*(GD+2) + 0] = 1.0*h*h;
        out[j*(GD+2) + 0] = 1.0*h*h;
        in[j*(GD+2) + GD+1] = 1.0*h*h;
        out[j*(GD+2) + GD+1] = 1.0*h*h;
    }
    for(int i = 0; i < GD+2; ++i)
    {
        in [0*(GD+2) + i] = -1.0*h*h;
        out[0*(GD+2) + i] = -1.0*h*h;
        in [(GD+1)*(GD+2) + i] = -1.0*h*h;
        out[(GD+1)*(GD+2) + i] = -1.0*h*h;
    }
    in [0*(GD+2) + 0] = 0.0;
    out[0*(GD+2) + 0] = 0.0;
    in [0*(GD+2) + (GD+1)] = 0.0;
    out[0*(GD+2) + (GD+1)] = 0.0;

    in [(GD+1)*(GD+2) + 0] = 0.0;
    out[(GD+1)*(GD+2) + 0] = 0.0;
    in [(GD+1)*(GD+2) + (GD+1)] = 0.0;
    out[(GD+1)*(GD+2) + (GD+1)] = 0.0;

    printf("Initial: %30.20le\n", L2norm(GD,  (double (*)[GD+2])  in));

    double start = omp_get_wtime();
    uint64_t start_c = _rdtsc();
    for(int t = 0; t < iterations; ++t)
    {
#ifdef USE_GEN
        inplace_laplace(GD, (double (*)[GD+2]) in, 1, GD+1, 1, GD+1);
#else
        test_compute(GD, (double (*)[GD+2]) in, (double (*)[GD+2]) out);
        std::swap(in, out);
#endif
    }

    double end = omp_get_wtime();
    uint64_t end_c = _rdtsc();
    printf("Took %le seconds\n", end-start);
    printf("Took %le cycles\n", (double)(end_c-start_c));

    const uint64_t total_cyc = end_c - start_c;
    printf("Took %le cycles/iter\n", (double)total_cyc/iterations);
    printf("Took %le cycles/iter/cell\n", (double)total_cyc/iterations/(GD*GD));

    printf("Final %30.20le\n", L2norm(GD,  (double (*)[GD+2]) in));
}
