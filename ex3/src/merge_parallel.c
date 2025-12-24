#include "merge_parallel.h"

#include<stdlib.h>

void parallel_merge(int *arr, long long l, long long m, long long r) {
    long long left = l;
    long long right = m + 1;
    long long length = r - l + 1;
    int *temp = malloc((size_t)length * sizeof(int));
    long long index = 0;

    /* start merging the two halves */
    while (left <= m && right <= r) {
        if (arr[left] < arr[right]) {
            temp[index++] = arr[left];
            left++;
        } else {
            temp[index++] = arr[right];
            right++;
        }
    }

    /* copy the rest to the temp array */
    while (left <= m) {
        temp[index++] = arr[left];
        left++;
    }
    while (right <= r) {
        temp[index++] = arr[right];
        right++;
    }

    /* copy the temp array back to the input array */
    for (long long i = 0; i < length; i++) {
        arr[l + i] = temp[i];
    }
}

void parallel_mergesort(int *arr, long long l, long long r) {
    if (l < r) {
        long long mid = (l + r) / 2;
        parallel_mergesort(arr, l, mid);
        parallel_mergesort(arr, mid + 1, r);
        parallel_merge(arr, l, mid, r);
    }
}