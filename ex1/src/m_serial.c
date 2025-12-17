// #define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// A is the first poly on n degree and B is the second poly of m degree (consider can be n = m)
long long  *m_serial(const long long *A, size_t n, const long long *B, size_t m, double *time) {
    
    size_t rdeg = n + m ;
    long long *R = calloc((size_t)rdeg + 1, sizeof(long long));
    if(!R) {
        perror("calloc");
        return NULL;
    }

    
    struct timespec begin,end;
    //unsigned int *seed = 0;
    //begining timestamp
    clock_gettime(CLOCK_MONOTONIC, &begin);

    //printf("Serial START\n");
    for (size_t i = 0; i <= n; i++){
        for(size_t j = 0; j <= m; j++){
            // __int128 tmp = (__int128)A[i] * (__int128)B[j] + (__int128)R[i + j];
            // R[i+j] = (long long) tmp;
            R[i+j] += A[i] * B[j];
        }
    }
    //time spent here approximately O(n^2)
    //getting ending timestamp
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_spent = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1e9;

    *time = time_spent;
    return R;
}