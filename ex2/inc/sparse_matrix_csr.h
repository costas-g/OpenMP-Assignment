#ifndef sparse_matrix_csr_h_
#define sparse_matrix_csr_h_

/* Struct that holds the pointers to the arrays of the CSR sparse matrix representation. 
 * Also holds the number of rows in the rows field. 
 * Fields: rows, values, col_index, row_ptr. 
 */
struct sparse_matrix_csr { 
    long long rows;
    int *values;
    long long *col_index;
    long long *row_ptr;
};

/* Creates a new sparse_matrix_csr object, initializes its fields, and returns it. Value fields are set to 0, and pointer fields to NULL. */
struct sparse_matrix_csr init_csr_matrix(void);

/* Buils the CSR sparse matrix representation of the input matrix. NNZ required. */
int build_csr_matrix(int **input_mtx, struct sparse_matrix_csr *output_mtx_csr, long long rows, long long cols, long long nnz);

/* Buils the CSR sparse matrix representation of the input matrix in parallel. NNZ required. */
int build_csr_matrix_parallel(int **input_mtx, struct sparse_matrix_csr *output_mtx_csr, long long rows, long long cols, long long nnz, size_t thread_count);

/* Frees the pointers associated with the sparse_matrix_csr struct. */
void free_csr_matrix(struct sparse_matrix_csr *mtx_csr);

/* Counts and returns the number of non-zero elements of a matrix. */
long long count_nnz(int **mtx, long long rows, long long cols);

/* Compares two sparse_matrix_csr structs. Returns 1 if they are the same, 0 if not. */
int compare_csr_matrix(struct sparse_matrix_csr *A, struct sparse_matrix_csr *B, long long nnz);

/* Print the CSR sparse matrix representation arrays. */
void print_csr_matrix(struct sparse_matrix_csr *M, long long nnz);

#endif