#include <stdio.h>
#include <stdlib.h>

#include "gen_sparse_matrix.h"

int **gen_sparse_matrix(long long rows, long long cols, size_t sparsity, long long *nnz, int max_val){
    /* Allocation for all the matrix elements, initialized to 0 */
    int *data = calloc((size_t)(rows * cols), sizeof(int));
    if (!data) {
        perror("malloc data");
        exit(EXIT_FAILURE);
    }

    /* Allocation of ROWS pointers to int pointers (row arrays) */
    int **mtx = malloc((size_t)(rows) * sizeof(int*));
    if (!mtx) {
        perror("malloc mtx");
        exit(EXIT_FAILURE);
    }

    /* Assign the row pointers */
    for (long long i = 0; i < rows; i++){
        mtx[i] = &data[i * cols];
    }

    /* Generate random values and count the number of non-zero elements */
    *nnz = 0;
    if (max_val < 2) max_val = RAND_MAX;
    for (long long i = 0; i < rows; i++){
        for (long long j = 0; j < cols; j++){
            /* Apply sparsity */
            if ((size_t)(rand() % 101) > sparsity){
                do 
                    mtx[i][j] = (rand() % max_val);// - max_val/2; /* in range [-max_val/2, max_val/2 - 1] */
                while (mtx[i][j] == 0);
                (*nnz)++;
            }
        }
    }
    
    return mtx;
}