#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "matvecs_csr.h"
#include "sparse_matrix_csr.h"
// #include "util_matvec.h"

void matvecs_csr(struct sparse_matrix_csr *A_csr, int *x, int *res, int iters){
    long long i, j;
    int r;
    long long rows = A_csr->rows;
    long long cols = rows; /* square matrix */

    if (iters < 1) {
        /* Copy input vector to output vector. */
        for (i = 0; i < cols; i++) {
            res[i] = x[i];
        }
        return;
    }

    int **x_tmp = malloc(2 * sizeof(int*));
    x_tmp[0] = malloc(2*cols * sizeof(int));
    x_tmp[1] = &x_tmp[0][cols];

    /* Copy input x vector to intermediate x_tmp vector. */
    for (i = 0; i < cols; i++) {
        x_tmp[0][i] = x[i];
    }
    
    int *x_curr;
    int *res_curr;

    #ifdef DEBUG
    double t_start, t_end;
    #endif
    

    for (r = 0; r < iters; r++) {
        x_curr = x_tmp[r % 2];
        res_curr = x_tmp[(r + 1) % 2];
        #ifdef DEBUG
        t_start = omp_get_wtime();
        #endif
        for (i = 0; i < rows; i++) {
            res_curr[i] = 0;
            for (j = A_csr->row_ptr[i]; j < A_csr->row_ptr[i+1]; j++) {
                res_curr[i] += A_csr->values[j] * x_curr[A_csr->col_index[j]];
            }
        }
        #ifdef DEBUG
        t_end = omp_get_wtime();
        printf("Sum loop time: %.7f s\n", t_end - t_start);
        #endif
    }

    /* Copy result to output memory */
    for (i = 0; i < cols; i++) {
        res[i] = res_curr[i];
    }

    /* Free allocated memory */
    free(x_tmp[0]);
    free(x_tmp);

    return;
}

void matvecs_csr_parallel(struct sparse_matrix_csr *A_csr, int *x, int *res, int iters, int thread_count) {
    long long rows = A_csr->rows;
    long long cols = rows; /* square matrix */

    if (iters < 1) {
        /* Copy input vector to output vector. */
        for (long long i = 0; i < cols; i++) {
            res[i] = x[i];
        }
        return;
    }

    /* global pointer that can point to all threads' intermediate results arrays 
     * thus every thread may write to any thread's intermediate result to complete its array before the next synced stage/iteration */
    // int ***res_currs = malloc(thread_count * sizeof(int**));

    /* Pointer that can point to two arrays. The two arrays will be use as intermediate result arrays 
     * In each stage/iteration, one is used as input to be read and the other gets written with the result. 
     * At every next stage, they are switched, so that the result array is now read as input and the input array is overwrritten with the new result. */
    int **x_tmp_global = malloc(2 * sizeof(int*));
    x_tmp_global[0] = malloc(2 * cols * sizeof(int)); /* allocate memory for the two arrays and assign them */
    x_tmp_global[1] = &x_tmp_global[0][cols]; /* assign the address of the second array to the pointer x_tmp_global[1] */

    # pragma omp parallel num_threads(thread_count)
    {
        // #ifdef _OPENMP
        // int tid = omp_get_thread_num();
        // int nthreads = omp_get_num_threads();
        // #else
        // int tid = 0;
        // int nthreads = 1;
        // #endif

        #ifdef DEBUG
        double t_start, t_end;
        #endif
        // t_start = omp_get_wtime();

        /* each thread assigns its intermediate results arrays to the global pointer */
        // res_currs[tid] = x_tmp;

        /* Copy input x vector to intermediate x_tmp_global vector. */
        # pragma omp single
        for (long long i = 0; i < cols; i++) {
            x_tmp_global[0][i] = x[i];
        }
        
        int *x_curr,    *res_curr = NULL; /* local, temporary pointers */
        int  x_curr_idx, res_curr_idx; /* index to select one of the two intermediate results arrays */
        // long long my_start_idx = -1, my_end_idx = -1; /* record each thread's start and end row */
        int sum;

        // #pragma omp barrier
        // #pragma omp single
        // {
        //     t_end = omp_get_wtime();
        //     printf("Mallocs time:  %.7f s\n", t_end - t_start);
        // }
        // t_start = omp_get_wtime();

        for (int r = 0; r < iters; r++) {
            /* one index will be 0 and the other will be 1, interchanged after each iteration */
            x_curr_idx   =      r  % 2;
            res_curr_idx = (r + 1) % 2;
            x_curr = x_tmp_global[x_curr_idx];     /* pointer assignment */
            res_curr = x_tmp_global[res_curr_idx]; /* pointer assignment */

            #ifdef DEBUG
            t_start = omp_get_wtime();
            #endif
            # pragma omp for schedule(static)
            for (long long i = 0; i < rows; i++) {
                // if(my_start_idx < 0) my_start_idx = i;  /* record the thread's start row */
                // my_end_idx = i + 1;                     /* record the thread's end row */
                sum = 0;
                for (long long j = A_csr->row_ptr[i]; j < A_csr->row_ptr[i+1]; j++) {
                    sum += A_csr->values[j] * x_curr[A_csr->col_index[j]];
                }
                res_curr[i] = sum;
            } /* implicit barrier */

            // #pragma omp barrier
            #ifdef DEBUG
            #pragma omp single
            {
                t_end = omp_get_wtime();
                printf("Sum loop time: %.7f s\n", t_end - t_start);
            }
            
            #endif

            // t_start = omp_get_wtime();
            /* Combine intermediate result */
            
            // THIS TAKES A LOT OF TIME ON LINUXLAB MACHINE. 
            // THIS INTRODUCES A LOT OF MEMORY TRAFFIC. 
            // MITIGATE BY USING ONLY ONE PAIR OF X_TMP ARRAYS AS GLOBAL. 
            // EACH THREAD READS FROM THE SAME GLOBAL ARRAY AND 
            // WRITES TO THE SAME GLOBAL ARRAY AT DIFFERENT LOCATIONS -> NO RACE CONDITION (EXCEPT FOR CACHE LINE OVERLAP FALSE SHARING). 
            // SWITCH AT THE START OF A NEW ITERATION.
            
            /* Write my partial result to the other threads' temp arrays */
            /* Shouldn't be a critical section - no two threads write to the same location */
            // for (int t = 0; t < nthreads; t++) {
            //     if (t != tid) {
            //         for (long long i = my_start_idx; i < my_end_idx; i++) {
            //             res_currs[t][res_curr_idx][i] = res_curr[i];
            //         }
            //     }
            // }
            // # pragma omp barrier /* need barrier to synchronize next stage */
            // #ifdef DEBUG
            // #pragma omp single
            // {
            //     t_end = omp_get_wtime();
            //     printf("Combine time:  %.7f s\n", t_end - t_start);
            // }
            // #endif
        }

        /* Copy final result to output memory */
        /* Shouldn't be a critical section */
        #ifdef DEBUG
            t_start = omp_get_wtime();
        #endif

        # pragma omp single
        for (long long i = 0; i < cols; i++) {
            res[i] = res_curr[i];
        }

        #ifdef DEBUG
        #pragma omp single
        {
            t_end = omp_get_wtime();
            printf("Final copy time:  %.7f s\n", t_end - t_start);
        }
        #endif
        // for (long long i = my_start_idx; i < my_end_idx; i++) {
        //     res[i] = res_curr[i];
        // }

        /* Free allocated memory */
        // free(x_tmp[0]);
        // free(x_tmp);
    }
    
    /* Free allocated memory */
    // free(res_currs);
    free(x_tmp_global[0]);
    free(x_tmp_global);

    return;
}