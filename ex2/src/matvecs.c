#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "matvecs.h"

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

    int **x_tmp = malloc(2 * sizeof(int*));
    x_tmp[0] = malloc(2*size * sizeof(int));
    x_tmp[1] = &x_tmp[0][size];

    /* Copy input x vector to intermediate x_tmp vector. */
    for (i = 0; i < size; i++) {
        x_tmp[0][i] = x[i];
    }
    
    int *x_curr;
    int *res_curr;

    for (r = 0; r < iters; r++) {
        x_curr = x_tmp[r % 2];
        res_curr = x_tmp[(r + 1) % 2];
        for (i = 0; i < size; i++) {
            res_curr[i] = 0;
            for (j = 0; j < size; j++) {
                res_curr[i] += A[i][j] * x_curr[j];
            }
        }
    }

    /* Copy result to output memory */
    for (i = 0; i < size; i++) {
        res[i] = res_curr[i];
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

    int ***res_currs = malloc(thread_count * sizeof(int**));

    # pragma omp parallel num_threads(thread_count)
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        #else
        int tid = 0;
        int nthreads = 1;
        #endif

        int **x_tmp = malloc(2 * sizeof(int*));
        x_tmp[0] = malloc(2 * size * sizeof(int));
        x_tmp[1] = &x_tmp[0][size];

        res_currs[tid] = x_tmp;

        /* Copy input x vector to intermediate x_tmp vector. */
        for (long long i = 0; i < size; i++) {
            x_tmp[0][i] = x[i];
        }
        
        int *x_curr,    *res_curr;
        int  x_curr_idx, res_curr_idx;
        long long my_start_idx = -1, my_end_idx = -1;

        for (int r = 0; r < iters; r++) {
            x_curr_idx   =      r  % 2;
            res_curr_idx = (r + 1) % 2;
            x_curr = x_tmp[x_curr_idx];
            res_curr = x_tmp[res_curr_idx];
            # pragma omp for 
            for (long long i = 0; i < size; i++) {
                if(my_start_idx < 0) my_start_idx = i;
                my_end_idx = i + 1;
                res_curr[i] = 0;
                for (long long j = 0; j < size; j++) {
                    res_curr[i] += A[i][j] * x_curr[j];
                }
            } /* implicit barrier */

            // # pragma omp critical 
            // {
            //     printf("T%d > res_curr                         = ", tid); print_vector(res_curr, size);
            //     printf("T%d > res_currs[tid=%d][res_curr_idx=%d] = ", tid, tid,res_curr_idx); print_vector(res_currs[tid][res_curr_idx], size);
            // }
            
            /* Combine intermediate result */
            /* Write my partial result to the other threads' temp arrays */
            /* Shouldn't be a critical section - no two threads write to the same location */
            for (int t = 0; t < nthreads; t++) {
                if (t != tid) {
                    for (long long i = my_start_idx; i < my_end_idx; i++) {
                        res_currs[t][res_curr_idx][i] = res_curr[i];
                    }
                }
            }
            # pragma omp barrier /* need barrier to synchronize next stage */

            // # pragma omp critical 
            // {
            //     printf("T%d > res_currs[tid=%d][res_curr_idx=%d] = ", tid, tid,res_curr_idx); print_vector(res_currs[tid][res_curr_idx], size);
            // }
        }

        /* Copy final result to output memory */
        /* Shouldn't be a critical section */
        for (long long i = my_start_idx; i < my_end_idx; i++) {
            res[i] = res_curr[i];
        }

        /* Free allocated memory */
        free(x_tmp[0]);
        free(x_tmp);
    }
    
    free(res_currs);

    return;
}

long long vectors_diffs(int* vec_a, int* vec_b, long long size) {
    long long num_errors = 0;
    for (long long i = 0; i < size; i++) {
        if (vec_a[i] != vec_b[i]) {
            num_errors++;
        }
    }
    return num_errors;
}

void print_matrix(int **mtx, long long rows, long long cols) {
    long long i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%d, ", mtx[i][j]);
        }
        puts("");
    }
}

void print_vector(int *vec, long long size) {
    long long i;
    for (i = 0; i < size; i++) {
        printf("%d, ", vec[i]);
    }
    puts("");
}