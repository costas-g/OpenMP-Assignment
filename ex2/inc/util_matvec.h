#ifndef util_matvec_h
#define util_matvec_h

#include <stdio.h>

static long long vectors_diffs(int* vec_a, const int* vec_b, long long size) {
    long long num_errors = 0;
    for (long long i = 0; i < size; i++) {
        if (vec_a[i] != vec_b[i]) {
            num_errors++;
        }
    }
    return num_errors;
}

#ifdef DEBUG
static void print_matrix(int **mtx, long long rows, long long cols) {
    long long i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%d, ", mtx[i][j]);
        }
        puts("");
    }
}
#endif

#ifdef DEBUG
static void print_vector(int *vec, long long size) {
    long long i;
    for (i = 0; i < size; i++) {
        printf("%d, ", vec[i]);
    }
    puts("");
}
#endif

#endif