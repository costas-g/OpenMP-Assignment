// #define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long  *m_serial(const long long *A, size_t n, const long long *B, size_t m, double *time) {
    size_t rdeg = n + m; /* resultant degree */
    long long *R = calloc((size_t)rdeg + 1, sizeof(long long));
    if(!R) {
        perror("calloc R");
        exit(EXIT_FAILURE);
    }

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    for (size_t i = 0; i <= n; i++){
        for(size_t j = 0; j <= m; j++){
            R[i+j] += A[i] * B[j];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end); /* end time */

    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    *time = time_spent;

    return R;
}