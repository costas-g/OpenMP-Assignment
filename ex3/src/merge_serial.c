#include "merge_serial.h"

#include<stdlib.h>

/* Textbook merge function using a pre-allocated temporary array */
void merge(int *arr, int *tmp, long long l, long long m, long long r) {
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

/* Textbook mergesort function using a pre-allocated temporary array */
void mergesort(int *arr, int *tmp, long long l, long long r) {
    if (l < r) {
        long long mid = l + (r - l) / 2; // (l + r) / 2;
        mergesort(arr, tmp, l, mid);
        mergesort(arr, tmp, mid + 1, r);
        merge(arr, tmp, l, mid, r);
    }
}