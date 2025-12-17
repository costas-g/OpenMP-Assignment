// #define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

long long  *m_parallel(const long long *A, size_t n, const long long *B, size_t m, size_t thread_count, double *time){

    // Initialization and declaration 
    struct timespec begin, end;
    size_t r = n + m + 1;
    

    long long *R_global = calloc(r, sizeof(long long)); 
    if (!R_global) {
        perror("calloc R_global");
        exit(EXIT_FAILURE);
    }

    long long **R_locals = malloc(thread_count * sizeof(*R_locals));
    if (!R_locals) {
        perror("malloc R_locals");
        exit(EXIT_FAILURE);
    }

    long long *R_local;
    long long prod;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    # pragma omp parallel num_threads(thread_count) private(R_local, prod)
    {   
        int tid = omp_get_thread_num();
        // Each thread allocates its private array
        R_locals[tid] = calloc(r, sizeof(long long));
        // if (!R_locals[tid]) {
        //     perror("calloc R_local");
        //     exit(EXIT_FAILURE);
        // }
        R_local = R_locals[tid];

        # pragma omp for nowait /* nowait removes the implicit barrier after the for block*/
        for (size_t i = 0; i <= n; i++){
            for(size_t j=0; j <= m ; j++){
                prod = A[i] * B[j];
                R_local[i+j] += prod;   // per-thread private memory
            }
        }

        /* Combine results */
        size_t k = tid*r/thread_count; /* start from evenly spaced out indexes so that contention is reduced */
        for (size_t i = 0; i < r; i++, k++){
            if (k >= r)
                k = 0; /* wrap around to the head of the array */
            // # pragma omp atomic /* for safely updating a shared variable */
            R_global[k] += R_local[k];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    for (size_t t = 0; t < thread_count; t++) {
        /* Combine results sequentially */
        // for (size_t i = 0; i < r; i++) {
        //     R_global[i] += R_locals[t][i];
        // }

        /* Free memory */
        free(R_locals[t]);
    }

    /* Free memory */
    free(R_locals);
    
    
    double time_spent = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1e9;
    *time = time_spent;

    return R_global;
}