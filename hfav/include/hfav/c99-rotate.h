#ifndef _HFAV_C99_ROTATE_H_
#define _HFAV_C99_ROTATE_H_

#include <limits.h>

#define __hfav_max(x, y) (((x) > (y)) ? (x) : (y))
#define __hfav_min(x, y) (((x) < (y)) ? (x) : (y))

typedef int int32;
typedef long int64;
typedef float float32;
typedef double float64;

#define ROTATE(T) \
static inline void rotate_##T(T v[], int start, int end, int s) \
{ \
    for (int i = start; i < end; ++i) \
    { \
        v[i] = v[i+s]; \
    } \
}

#define VROTATE(T) \
static inline void vrotate_##T(T v[][VLEN], int start, int end, int s) \
{ \
    for (int i = start; i < end; ++i) \
    { \
        _Pragma("simd assert") \
        for (int j = 0; j < VLEN; ++j) \
        { \
            v[i][j] = v[i+s][j]; \
        } \
    } \
}

#define ROTATE_PTR(T) \
static inline void rotate_##T##_ptr(T* v[], int len) \
{ \
    T* temp = v[0]; \
    for (int i = 0; i < len-1; ++i) \
    { \
        v[i] = v[i+1]; \
    } \
    v[len-1] = temp; \
}

ROTATE(int32)
ROTATE(int64)
ROTATE(float32)
ROTATE(float64)

VROTATE(int32)
VROTATE(int64)
VROTATE(float32)
VROTATE(float64)

ROTATE_PTR(int32)
ROTATE_PTR(int64)
ROTATE_PTR(float32)
ROTATE_PTR(float64)

#endif /* _HFAV_C99_ROTATE_H_ */
