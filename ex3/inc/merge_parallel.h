#ifndef merge_parallel_h
#define merge_parallel_h

// #define TASK_CUTOFF 100000
// #include <stddef.h> /* defines size_t */

void parallel_merge(int *arr, int *tmp, long long l, long long m, long long r);
void parallel_mergesort(int *arr, int *tmp, long long l, long long r);
void start_parallel_mergesort(int *arr, int *tmp, long long l, long long r, size_t thread_count);

#endif