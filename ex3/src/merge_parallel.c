#include<stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "merge_parallel.h"
#include "merge_serial.h" /* for the merge function */

/* The same sequential merge function is used */

#include <stdio.h>
int DEBUG = 0;

long long TASK_CUTOFF; /* subarray size below which no further tasks are created */

/* Recursive parallel mergesort algorthm using tasks */
void parallel_mergesort(int *arr, int *tmp, long long l, long long r) {
    // printf("Thread %d > mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), l, r);
    if (l < r) {
        long long mid = l + (r - l) / 2; // (l + r) / 2
        long long curr_size = r - l + 1;

        int task_enable = curr_size > TASK_CUTOFF;

        if (task_enable && DEBUG) printf("Thread %d > Task mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), l, mid);
        # pragma omp task if (task_enable) shared(arr, tmp) firstprivate(l, mid)
        {
            if (task_enable && DEBUG) printf("Thread %d > mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), l, mid);
            parallel_mergesort(arr, tmp, l, mid);
        }

        if (task_enable && DEBUG) printf("Thread %d > Task mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), mid + 1, r);
        # pragma omp task if (task_enable) shared(arr, tmp) firstprivate(mid, r)
        {
            if (task_enable && DEBUG) printf("Thread %d > mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), mid + 1, r);
            parallel_mergesort(arr, tmp, mid + 1, r);
        }
        
        # pragma omp taskwait
        merge(arr, tmp, l, mid, r);
    }
    // printf("Thread %d > mergesort(arr, %lld, %lld) finished\n", omp_get_thread_num(), l, r);
}

/* Entry point for the parallel mergesort function */
void begin_parallel_mergesort(int *arr, int *tmp, long long l, long long r, size_t thread_count) {
    long long size = r - l + 1; /* size of the full array */
    size_t avg_tasks_per_thread = 4; /* 2-8? */
    TASK_CUTOFF = size/(avg_tasks_per_thread * thread_count);
    /* or */
    size_t max_level = 2; /* max depth level to spawn new tasks */
    TASK_CUTOFF = size/(1u << max_level);
    /* or */
    // TASK_CUTOFF = 20000000;

    # pragma omp parallel num_threads(thread_count)
    {
        # pragma omp single
        {
            if (DEBUG) printf("Thread %d > mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), l, r);
            parallel_mergesort(arr, tmp, l, r);
        }
    }
}