#include <stdio.h>
#include <stdlib.h>

#include "sparse_matrix_csr.h"

struct sparse_matrix_csr init_csr_matrix(void) {
    struct sparse_matrix_csr m = {
        .rows       = 0, 
        .values     = NULL, 
        .col_index  = NULL, 
        .row_ptr    = NULL
    };
    return m;
}

void build_csr_matrix(int **input_mtx, struct sparse_matrix_csr *output_mtx_csr, long long rows, long long cols, long long nnz){
    struct sparse_matrix_csr *csr = output_mtx_csr;
    csr->rows = rows;
    csr->values    = malloc( nnz * sizeof(int));
    csr->col_index = malloc( nnz * sizeof(long long));
    csr->row_ptr   = malloc( (rows+1) * sizeof(long long));

    int val;
    long long idx = 0;
    csr->row_ptr[0] = 0;

    for (long long i = 0; i < rows; i++) {
        csr->row_ptr[i+1] = csr->row_ptr[i];
        for (long long j = 0; j < cols; j++) {
            val = input_mtx[i][j];
            if (val){
                // add check if idx > nnz-1
                csr->values[idx] = val;
                csr->col_index[idx] = j;
                idx++;
                csr->row_ptr[i+1]++;
            }
        }
    }

    if (csr->row_ptr[rows] == nnz)
        printf("  CSR representation built successfully. NNZ = %lld\n", csr->row_ptr[rows]);
    else 
        printf("  ERROR: Mismatch. Given NNZ = %lld. Counted NNZ = %lld\n", nnz, csr->row_ptr[rows]);

    return;
};

void build_csr_matrix_parallel(int **input_mtx, struct sparse_matrix_csr *output_mtx_csr, long long rows, long long cols, long long nnz, size_t thread_count){
    struct sparse_matrix_csr *csr = output_mtx_csr;

    if (!csr->row_ptr) return;
    
    if (csr->row_ptr[rows] == nnz)
        printf("  CSR representation built successfully. NNZ = %lld\n", csr->row_ptr[rows]);
    else 
        printf("  ERROR: Mismatch. Given NNZ = %lld. Counted NNZ = %lld\n", nnz, csr->row_ptr[rows]);
    
    return;
}

void free_csr_matrix(struct sparse_matrix_csr *mtx_csr){
    free(mtx_csr->values);
    free(mtx_csr->col_index);
    free(mtx_csr->row_ptr);

    return;
}

int compare_csr_matrix(struct sparse_matrix_csr *A, struct sparse_matrix_csr *B, long long nnz){
    if (A->rows != B->rows)
        return 0;

    for (long long i = 0; i < nnz; i++){
        if (A->values[i] != B->values[i]){
            return 0;
        }
        if (A->col_index[i] != B->col_index[i]){
            return 0;
        }
    }

    for (long long i = 0; i < A->rows+1; i++){
        if (A->row_ptr[i] != B->row_ptr[i]){
            return 0;
        }
    }

    return 1;
}

void print_csr_matrix(struct sparse_matrix_csr *M, long long nnz){
    printf("\nPrinting CSR matrix...\n");
    printf("  NNZ = %lld\n", nnz);

    if (M->values != NULL){
        printf("  values  = [");
        for (long long i = 0; i < nnz; i++){
            printf("%d, ", M->values[i]);
        }
        printf("]\n");
    }
        
    if (M->col_index != NULL){
        printf("  col_ind = [");
        for (long long i = 0; i < nnz; i++){
            printf("%lld, ", M->col_index[i]);
        }
        printf("]\n");
    }

    if (M->row_ptr != NULL){
        printf("  row_ptr = [");
        for (long long i = 0; i < M->rows+1; i++){
            printf("%lld, ", M->row_ptr[i]);
        }
        printf("]\n");
    }
    
    return;
}