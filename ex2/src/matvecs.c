#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "matvecs.h"
// #include "util_matvec.h"

void matvecs(int **A, int *x, int *res, long long size, int iters){
    long long i, j;
    int r;

    if (iters < 1) {
        /* Copy input vector to output vector. */
        for (i = 0; i < size; i++) {
            res[i] = x[i];
        }
        return;
    }

    /* Pointer that can point to two arrays. The two arrays will be used as intermediate result arrays 
     * In each stage/iteration, one is used as input to be read and the other gets written with the result. 
     * At every next stage, they are switched, so that the result array is now read as input and the input array is overwrritten with the new result.  */
    int **x_tmp = malloc(2 * sizeof(int*));
    x_tmp[0] = malloc(2*size * sizeof(int)); /* allocate memory for the two arrays and assign them */
    x_tmp[1] = &x_tmp[0][size]; /* assign the address of the second array to the pointer x_tmp_global[1] */

    /* Copy input x vector to intermediate x_tmp vector. */
    for (i = 0; i < size; i++) {
        x_tmp[0][i] = x[i];
    }
    
    int *x_read;
    int *x_write;

    for (r = 0; r < iters; r++) {
        x_read = x_tmp[r % 2];
        x_write = x_tmp[(r + 1) % 2];
        for (i = 0; i < size; i++) {
            x_write[i] = 0;
            for (j = 0; j < size; j++) {
                x_write[i] += A[i][j] * x_read[j];
            }
        }
    }

    /* Copy result to output memory */
    for (i = 0; i < size; i++) {
        res[i] = x_write[i];
    }

    /* Free allocated memory */
    free(x_tmp[0]);
    free(x_tmp);

    return;
}

void matvecs_parallel(int **A, int *x, int *res, long long size, int iters, int thread_count) {
    if (iters < 1) {
        /* Copy input vector to output vector. */
        for (long long i = 0; i < size; i++) {
            res[i] = x[i];
        }
        return;
    }

    /* Global pointer that can point to two arrays. The two arrays will be used as intermediate result arrays 
     * In each stage/iteration, one is used as input to be read and the other gets written with the result. 
     * At every next stage, they are switched, so that the result array is now read as input and the input array is overwrritten with the new result.  */
    int **x_tmp_global = malloc(2 * sizeof(int*));
    x_tmp_global[0] = malloc(2 * size * sizeof(int)); /* allocate memory for the two arrays and assign them */
    x_tmp_global[1] = &x_tmp_global[0][size]; /* assign the address of the second array to the pointer x_tmp_global[1] */

    # pragma omp parallel num_threads(thread_count)
    {
        // #ifdef _OPENMP
        // int tid = omp_get_thread_num();
        // int nthreads = omp_get_num_threads();
        // #else
        // int tid = 0;
        // int nthreads = 1;
        // #endif

        /* Copy input x vector to intermediate x_tmp_global vector. */
        # pragma omp single /* omp single: better when size is little, omp for: better when size is a lot */
        for (long long i = 0; i < size; i++) {
            x_tmp_global[0][i] = x[i]; /* (in case of parallel for) not a critical section because of different memory locations */
        }
        
        int *x_read = NULL, *x_write = NULL;    /* local, temporary pointers */
        int  x_read_idx,     x_write_idx;       /* index to select one of the two intermediate results arrays */
        int sum; /* private temporary variable */

        for (int r = 0; r < iters; r++) {
            /* one index will be 0 and the other will be 1, interchanged after each iteration */
            x_read_idx  =      r  % 2;
            x_write_idx = (r + 1) % 2;
            x_read  = x_tmp_global[x_read_idx];  /* pointer assignment */
            x_write = x_tmp_global[x_write_idx]; /* pointer assignment */

            # pragma omp for schedule(static)
            for (long long i = 0; i < size; i++) {
                sum = 0;
                for (long long j = 0; j < size; j++) {
                    sum += A[i][j] * x_read[j];
                }
                x_write[i] = sum; /* not a critical section because of different memory locations */
            } /* implicit barrier */

            /* need barrier to synchronize next stage */
            /* no need to declare if nothing happens after the last implicit barrier */
            // # pragma omp barrier 
        }

        /* Copy final result to output memory */
        # pragma omp single /* omp single: better when cols are few, omp for: better when cols are a lot */
        for (long long i = 0; i < size; i++) {
            res[i] = x_write[i]; /* (in case of parallel for) not a critical section because of different memory locations */
        }
    }
    
    /* Free allocated memory */
    free(x_tmp_global[0]);
    free(x_tmp_global);

    return;
}