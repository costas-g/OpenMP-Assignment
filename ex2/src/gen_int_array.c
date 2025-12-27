#include <stdio.h>
#include <stdlib.h>

#include "gen_int_array.h"

int *gen_int_array(long long size, int max_val){
    int *arr = malloc((size_t)(size) * sizeof(int));
    if (!arr) {
        perror("malloc arr");
        exit(EXIT_FAILURE);
    }

    // max_val = RAND_MAX;
    for (long long i = 0; i < size; i++){
        arr[i] = (rand() % max_val) - max_val/2; /* in range [-max_val/2, max_val/2 - 1] */
    }

    return arr;
}