/* examples/hydro2d/array-macros.hpp: utilities for HPC codes

   (C) Jason Sewall : Intel -- initial version
   (C) John Pennycook : Intel -- augmentations to above version
*/
/*
  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.
*/

#ifndef _ARRAY_MACROS_HPP__
#define _ARRAY_MACROS_HPP__

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <fcntl.h>

__attribute__((noreturn))
static inline void die(const char *fmt, ...)
{
    va_list val;
    va_start(val, fmt);
    vfprintf(stderr, fmt, val);
    va_end(val);
    exit(EXIT_FAILURE);
}

typedef unsigned long long u64;

#define CACHE_LINE_BYTES 64

inline void divvy(u64 *start, u64 *end, const u64 nitems, u64 chunkno, u64 nchunks)
{
    const u64 items_per_chunk = nitems/nchunks;
    const u64 remainder       = nitems - nchunks*items_per_chunk;

    *start = chunkno*items_per_chunk     + std::min(chunkno,   remainder);
    *end   = (chunkno+1)*items_per_chunk + std::min(chunkno+1, remainder);
}

inline unsigned long long round_to_alignment(unsigned long long x, int alignment)
{
    if(x & (alignment-1))
        x = (x & ~(alignment-1)) + alignment;
    return x;
}


inline void *aligned_malloc(size_t bytes)
{
    void *ptr;
    if(posix_memalign(&ptr, CACHE_LINE_BYTES, bytes))
        return 0;
    return ptr;
}

inline void aligned_free(void *ptr)
{
    free(ptr);
}

static void *xmalloc(size_t sze, const char *name)
{
    void *res = malloc(sze);
    if(!res)
        die("Failed to allocate %zub for %s!\n", sze, name);
    return res;
}

static void xmalloc_free(void *p)
{
    free(p);
}

static void *xaligned_malloc(size_t sze, const char *name)
{
    void *res = aligned_malloc(sze);
    if(!res)
        die("Failed to allocate %zub for %s!\n", sze, name);
    return res;
}

static void xaligned_malloc_free(void *p)
{
    free(p);
}

static void *xrealloc(void *ptr, size_t sze, const char *name)
{
    void *res = realloc(ptr, sze);
    if(!res)
        die("Failed to allocate %zub for %s!\n", sze, name);
    return res;
}

static char *xstrdup(const char *ptr, const char *name)
{
    char *res = strdup(ptr);
    if(!res)
        die("Failed to strdup %zub for %s!\n", strlen(ptr), name);
    return res;
}

static void xstrdup_free(void *ptr)
{
    free(ptr);
}

static char *xstrndup(const char *ptr, size_t n, const char *name)
{
    char *res = strndup(ptr,n );
    if(!res)
        die("Failed to strndup %zub for %s!\n", strlen(ptr), name);
    return res;
}

static int xsnprintf(char *str, size_t n, const char *fmt, ...)
{
    va_list val;
    va_start(val, fmt);
    int wanted_out = vsnprintf(str, n, fmt, val);
    va_end(val);
    if(wanted_out > n)
    {
        die("Ran out of buffer space for output string!\n");
    }
    return wanted_out;
}

static FILE *xfopen_write(const char *str, char *mode)
{
    int fd = creat(str, S_IRUSR | S_IRGRP | S_IWUSR | S_IWGRP);
    if(fd == -1)
        return 0;

    FILE *file = fdopen(fd, mode);
    return file;
}

static FILE *xfopen_read(const char *str, char *mode)
{
    int fd = open(str, O_RDONLY);
    if(fd == -1)
        return 0;

    FILE *file = fdopen(fd, mode);
    return file;
}

static char *human_format(double in)
{
    static const char cf_chars[]              = {'t', 'g', 'm', 'k', 0};
    static const unsigned long long cf_vals[] = {
        1ULL << 40,
        1ULL << 30,
        1ULL << 20,
        1ULL << 10,
        0ULL,
    };
    const double     ain     = std::abs(in);

    int i;
    for(i = 0; ain < cf_vals[i]; ++i);

    double v = in/std::max(cf_vals[i], 1ULL);
    char buff[1024];
    xsnprintf(buff, 1023, "%.1lf%c", v, cf_chars[i]);
    return strdup(buff);
}

static long long suffixed_atoll(const char *nptr, int nthreads)
{
    char      *mod;
    double     mul = strtod(nptr, &mod);
    while(*mod)
    {
        switch(*mod)
        {
        case 't':
            mul *= nthreads;
            break;
        case 'T':
            mul *= nthreads;
            break;
        case 'k':
            mul *= 1024;
            break;
        case 'K':
            mul *= 1000;
            break;
        case 'm':
                mul *= 1024*1024;
                break;
            case 'M':
                mul *= 1000000;
                break;
            case 'g':
                mul *= 1024*1024*1024;
                break;
            case 'G':
                mul *= 1000000000;
                break;
            default:
                return mul;
            }
            ++mod;
        }
        return mul;
    }

    #define DECLARE_ARRAY_ALL(type, name)       \
    int name##_n;                               \
    int name##_n_allocd;                        \
    type* name

#define INIT_ARRAY(name, size)                  \
    name##_n = 0;                               \
    name##_n_allocd = size;                     \
    name     = (typeof(name)) xmalloc(sizeof(name[0])*name##_n_allocd, #name);

#define INIT_ARRAY_ALIGNED(name, size)          \
    name##_n        = 0;                        \
    name##_n_allocd = size;                     \
    name            = (typeof(name)) xaligned_malloc(sizeof(name[0])*name##_n_allocd, #name);

#define EXTEND_ARRAY(name, num)                 \
    if(name##_n + num >= name##_n_allocd)       \
    {                                           \
        name##_n_allocd = (name##_n + num)*2;   \
        void *m         = xrealloc(name, sizeof(name[0])*name##_n_allocd, #name); \
        name            = (typeof(name)) m;                             \
    }

#define EXTEND_ARRAY_ALIGNED(name, num)         \
    if(name##_n + num >= name##_n_allocd)       \
    {                                           \
        name##_n_allocd = (name##_n + num)*2;   \
        void *m         = xaligned_malloc(sizeof(name[0])*name##_n_allocd, #name); \
        memcpy(m, name, sizeof(name[0])*name##_n);      \
        aligned_free(name);                             \
        name            = (typeof(name)) m;             \
    }

#define FREE_ARRAY_ALL(name)             \
    name##_n        = 0;                 \
    name##_n_allocd = 0;                 \
    free(name);                          \
    name            = 0;

#define FREE_ARRAY(name)                 \
    name##_n = 0;                        \
    free(name);                          \
    name     = 0;

#define FREE_ARRAY_ALIGNED(name)                \
    name##_n = 0;                               \
    aligned_free(name);                         \
    name     = 0;

#endif
