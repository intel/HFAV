#ifndef _HFAV_CPP_ROTATE_H_
#define _HFAV_CPP_ROTATE_H_

namespace hfav
{

template <typename T>
static inline void rotate(T v[], int start, int end, int s)
{
    for (int i = start; i < end; ++i)
    {
        v[i] = v[i+s];
    }
}

template <typename T>
static inline void vrotate(T v[][VLEN], int start, int end, int s)
{
    for (int i = start; i < end; ++i)
    {
        #pragma simd assert
        for (int j = 0; j < VLEN; ++j)
        {
            v[i][j] = v[i+s][j];
        }
    }
}

template <typename T>
static inline void rotate_ptr(T v[], int len)
{
    const T temp = v[0];
    for(int i = 0; i < len-1; ++i)
    {
        v[i] = v[i+1];
    }
    v[len-1] = temp;
}

}

#endif /* _HFAV_CPP_ROTATE_H_ */
