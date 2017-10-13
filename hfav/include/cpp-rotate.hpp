namespace hfav
{
    template <typename T>
    static inline void rotate(T v[], int len)
    {
        for(int i = 0; i < len-1; ++i)
            v[i] = v[i+1];
    }

    template <typename T>
    static inline void rotate_ptr(T v[], int len)
    {
        const T temp = v[0];
        for(int i = 0; i < len-1; ++i)
            v[i] = v[i+1];
        v[len-1] = temp;
    }
}
