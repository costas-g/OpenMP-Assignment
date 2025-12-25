#include<stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "merge_parallel.h"

long long TASK_CUTOFF;

void parallel_merge(int *arr, int *tmp, long long l, long long m, long long r) {
    long long left = l;         /* index for left half */
    long long right = m + 1;    /* index for right half */
    long long index = l;        /* index for tmp array */

    /* start sorting the two halves into tmp */
    while (left <= m && right <= r) {
        if (arr[left] < arr[right]) {
            tmp[index++] = arr[left];
            left++;
        } else {
            tmp[index++] = arr[right];
            right++;
        }
    }

    /* copy rest from left half into tmp array */
    while (left <= m) {
        tmp[index++] = arr[left];
        left++;
    }
    /* or copy rest from right half into tmp array */
    while (right <= r) {
        tmp[index++] = arr[right];
        right++;
    }

    /* copy sorted tmp array back into input array */
    for (long long i = l; i <= r; i++) {
        arr[i] = tmp[i];
    }
}
#include <stdio.h>
void parallel_mergesort(int *arr, int *tmp, long long l, long long r) {
    // printf("Thread %d > mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), l, r);
    if (l < r) {
        long long mid = l + (r - l) / 2; // (l + r) / 2
        long long curr_size = r - l + 1;

        int task_enable = curr_size > TASK_CUTOFF;

        // printf("Thread %d > Task mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), l, mid);
        # pragma omp task if (task_enable) shared(arr, tmp) firstprivate(l, mid)
        {
            // if (task_enable) printf("Thread %d > mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), l, mid);
            parallel_mergesort(arr, tmp, l, mid);
        }
    
        // printf("Thread %d > Task mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), mid + 1, r);
        # pragma omp task if (task_enable) shared(arr, tmp) firstprivate(mid, r)
        {
            // if (task_enable) printf("Thread %d > mergesort(arr, %lld, %lld)\n", omp_get_thread_num(), mid + 1, r);
            parallel_mergesort(arr, tmp, mid + 1, r);
        }
        
        # pragma omp taskwait
        parallel_merge(arr, tmp, l, mid, r);
    }
    // printf("Thread %d > mergesort(arr, %lld, %lld) finished\n", omp_get_thread_num(), l, r);
}

void start_parallel_mergesort(int *arr, int *tmp, long long l, long long r, size_t thread_count) {
    long long size = r - l + 1;
    size_t avg_tasks_per_thread = 64; /* 2-8? */
    TASK_CUTOFF = size/(avg_tasks_per_thread * thread_count);

    # pragma omp parallel num_threads(thread_count)
    {
        # pragma omp single 
        parallel_mergesort(arr, tmp, l, r);
    }
}