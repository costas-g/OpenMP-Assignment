#include <stdio.h>
#include <stdlib.h>

#include "gen_rand_int_array.h"

/* Allocates an array of SIZE integers, fills it with random values, and returns a pointer to it */
int *gen_rand_int_array(long long size){
    int *arr = malloc((size_t)(size) * sizeof(int));
    if (!arr) {
        perror("malloc arr");
        exit(EXIT_FAILURE);
    }

    for (long long i = 0; i < size; i++){
        arr[i] = (rand() % RAND_MAX) - RAND_MAX/2; /* in range [-RAND_MAX/2, RAND_MAX/2 - 1] */
    }

    return arr;
}