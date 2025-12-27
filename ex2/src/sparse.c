#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gen_rand_int_array.h"
#include "gen_sparse_matrix.h"
#include "sparse_matrix_csr.h"

void Usage(char* prog_name);

int main(int argc, char* argv[]) {
    long long matrix_size;  /* row/columnn size of square matrix */
    int sparsity;           /* percentage of zero-elements of the matrix, expressed */
    int num_mults;          /* number of repeated multiplications */
    int thread_count;

    /* Parse inputs and error check */
    if (argc < 5) Usage(argv[0]);

    matrix_size  = strtoll(argv[1], NULL, 10); if (matrix_size  <= 0) Usage(argv[0]);
    sparsity     =  strtol(argv[2], NULL, 10); if (sparsity     <= 0) Usage(argv[0]);
    num_mults    =  strtol(argv[3], NULL, 10); if (num_mults    <= 0) Usage(argv[0]);
    thread_count =  strtol(argv[4], NULL, 10); if (thread_count <= 0) Usage(argv[0]);

    long long rows = matrix_size, cols = matrix_size;

    printf("Square Matrix of dimensions NxN with N=%lld, sparsity=%d.\nRepeated multiplications: %d\nThread count: %d\n", matrix_size, sparsity, num_mults, thread_count);
    
    /* Timing variables */
    struct timespec start, end;
    double elapsed_time, gen_time;
    srand((unsigned int) time(NULL)); /* seed random generator */

    /* -------------------- Generate the matrix and the array of integers ---------------------- */
    printf("\nGenerating the square matrix of integers...\n");
    int **mtx;      /* pointer to the matrix of integers (like an array of int pointers)*/
    long long nnz;  /* number of non-zero elements generated */
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
            mtx = gen_sparse_matrix(matrix_size, matrix_size, sparsity, &nnz);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    gen_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; /* elapsed time */
    printf("  Matrix generate Time (s): %9.6f\n", gen_time);

    printf("\nGenerating the vector array of integers...\n");
    int *vec; /* pointer to the vector array of integers */
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
            vec = gen_rand_int_array(matrix_size);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    gen_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; /* elapsed time */
    printf("  Vector generate Time (s): %9.6f\n", gen_time);


    /* ----------------------- Build CSR Representation ----------------------- */

    struct sparse_matrix_csr *mtx_csr_ptr = malloc(sizeof(*mtx_csr_ptr));
    struct sparse_matrix_csr *mtx_csr_parallel_ptr = malloc(sizeof(*mtx_csr_parallel_ptr));
    *mtx_csr_ptr = init_csr_matrix();
    *mtx_csr_parallel_ptr = init_csr_matrix();

    /* Serial CSR Build */ 
    printf("\nSerial CSR build...\n");
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    build_csr_matrix(mtx, mtx_csr_ptr, rows, cols, nnz);
    clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; 
    printf("  Serial CSR build time (s):   %9.6f\n", elapsed_time);

    /* Parallel CSR Build */ 
    printf("\nParallel CSR build...\n");
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    build_csr_matrix_parallel(mtx, mtx_csr_parallel_ptr, rows, cols, nnz, (size_t) thread_count);
    clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; 
    printf("  Parallel CSR build time (s):   %9.6f\n", elapsed_time);

    /* ----------------------- Confirm CSR building correctness ----------------------- */
    printf("\nChecking CSR Build correctness...\n");
    int correct_build = compare_csr_matrix(mtx_csr_ptr, mtx_csr_parallel_ptr, nnz);

    if (correct_build) {
        printf("  Correct CSR build!\n");
    } else {
        printf("  ERROR: Incorrect CSR build!\n");
    }

    print_csr_matrix(mtx_csr_ptr, nnz);
    print_csr_matrix(mtx_csr_parallel_ptr, nnz);

    /* Free allocated memory */
    free(mtx[0]); // frees the contiguous data block
    free(mtx);
    free(vec);
    free_csr_matrix(mtx_csr_ptr);
    free_csr_matrix(mtx_csr_parallel_ptr);

    return 0;
} /* main */

/*--------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Print a message indicating how program should be started
 *            and terminate.
 */
void Usage(char *prog_name) {
   fprintf(stderr, "Usage: %s <matrix_size> <sparsity> <num_mults> <thread_count>\n", prog_name);
   fprintf(stderr, "   matrix_size: Row/column size (square matrix). Should be positive.\n");
   fprintf(stderr, "   sparsity: Percentage of zero-elements. Should be an integer from 0 up to 100.\n");
   fprintf(stderr, "   num_mults: Number of repeated multiplications. Should be positive.\n");
   fprintf(stderr, "   thread_count: Number of threads. Should be positive.\n");
   exit(0);
}  /* Usage */