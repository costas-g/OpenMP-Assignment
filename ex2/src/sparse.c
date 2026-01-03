#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gen_int_array.h"
#include "gen_sparse_matrix.h"
#include "sparse_matrix_csr.h"
#include "matvecs.h"
#include "matvecs_csr.h"
#include "util_matvec.h"

void Usage(char* prog_name);

int main(int argc, char* argv[]) {
    long long matrix_size;  /* row/columnn size of square matrix */
    float sparsity;         /* percentage of zero-elements of the matrix */
    int num_mults;          /* number of repeated multiplications */
    int thread_count;

    /* Parse inputs and error check */
    if (argc < 5) Usage(argv[0]);

    matrix_size  = strtoll(argv[1], NULL, 10); if (matrix_size  <= 0) Usage(argv[0]);
    sparsity     =  strtof(argv[2], NULL);     if (sparsity     <  0 || sparsity >= 1) Usage(argv[0]);
    num_mults    =  strtol(argv[3], NULL, 10); if (num_mults    <  0) Usage(argv[0]);
    thread_count =  strtol(argv[4], NULL, 10); if (thread_count <= 0) Usage(argv[0]);

    long long rows = matrix_size, cols = matrix_size;

    printf("Square Matrix of dimensions NxN with N=%lld, sparsity=%f\nRepeated multiplications: %d\nThread count: %d\n", matrix_size, sparsity, num_mults, thread_count);
    
    /* Timing variables */
    struct timespec start, end;
    double elapsed_time, gen_time;
    srand((unsigned int) time(NULL)); /* seed random generator */
    struct xorshift32_state prng_state;
    prng_state.a = (unsigned int) time(NULL); /* seed the PRNG */

    printf("\n================================================");
    /* -------------------- Generate the matrix and the array of integers ---------------------- */
    printf("\nGenerating the square matrix of integers...\n");
    int **mtx_p;      /* pointer to the matrix of integers (like an array of int pointers)*/
    long long nnz;  /* number of non-zero elements generated */
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
            mtx_p = gen_sparse_matrix(rows, cols, sparsity, 10, thread_count, &prng_state, &nnz);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    gen_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; /* elapsed time */
    printf("  Matrix generation time (s): %9.6f\n", gen_time);
    printf("  NNZ generated: %lld\n", nnz);

    printf("\nGenerating the vector array of integers...\n");
    int *vec; /* pointer to the vector array of integers */
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
            vec = gen_int_array(cols, 10);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    gen_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; /* elapsed time */
    printf("  Vector generation time (s): %9.6f\n", gen_time);

    
    /* ----------------------------- Build CSR Representation ----------------------------- */
    printf("\n================================================");
    struct sparse_matrix_csr *mtx_csr_ptr          = malloc(sizeof(struct sparse_matrix_csr));
    struct sparse_matrix_csr *mtx_csr_parallel_ptr = malloc(sizeof(struct sparse_matrix_csr));
    *mtx_csr_ptr = init_csr_matrix();
    *mtx_csr_parallel_ptr = init_csr_matrix();

    /* Serial CSR Build */ 
    printf("\nSerial CSR build...\n");
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    build_csr_matrix(mtx_p, mtx_csr_ptr, rows, cols, nnz);
    clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; 
    printf("  Serial CSR build time (s):   %9.6f\n", elapsed_time);

    /* Parallel CSR Build */ 
    printf("\nParallel CSR build...\n");
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    build_csr_matrix_parallel(mtx_p, mtx_csr_parallel_ptr, rows, cols, nnz, (size_t) thread_count);
    clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; 
    printf("  Parallel CSR build time (s): %9.6f\n", elapsed_time);

    /* Confirm CSR building correctness */
    printf("\nComparing Serial & Parallel CSR builds...\n");
    int correct_build = compare_csr_matrix(mtx_csr_ptr, mtx_csr_parallel_ptr, nnz);

    if (correct_build) {
        printf("  CSR builds match!\n");
    } else {
        printf("  ERROR: CSR builds don't match!\n");
    }

    // print_csr_matrix(mtx_csr_ptr, nnz);
    // print_csr_matrix(mtx_csr_parallel_ptr, nnz);


    /* -------------------- Dense matrix repeated multiplication ---------------------- */
    printf("\n================================================");
    int *vec_res          = malloc(rows * sizeof(int));
    int *vec_res_parallel = malloc(rows * sizeof(int));
    printf("\nDense matrix repeated multiplication SERIAL...\n");
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
            matvecs(mtx_p, vec, vec_res, matrix_size, num_mults);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; /* elapsed time */
    printf("  Dense matrix %dx mult Serial time (s):   %9.6f\n", num_mults, elapsed_time);
    // print_matrix(mtx_p, rows, cols);
    // print_vector(vec, rows);
    // print_vector(vec_res, rows);
    printf("\nDense matrix repeated multiplication PARALLEL...\n");
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
            matvecs_parallel(mtx_p, vec, vec_res_parallel, matrix_size, num_mults, thread_count);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; /* elapsed time */
    printf("  Dense matrix %dx mult Parallel time (s): %9.6f\n", num_mults, elapsed_time);
    // print_vector(vec_res_parallel, rows);

    /* Compare the two resulting vectors */
    printf("\nComparing Serial & Parallel results...\n");
    long long nerrors = vectors_diffs(vec_res, vec_res_parallel, matrix_size);
    if (nerrors == 0) {
        printf("  Results match!\n");
    } else {
        printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
    }


    /* -------------------- Sparse matrix repeated multiplication ---------------------- */
    printf("\n================================================");
    int *vec_res_sparse          = malloc(rows * sizeof(int));
    int *vec_res_sparse_parallel = malloc(rows * sizeof(int));
    printf("\nSparse matrix repeated multiplication SERIAL...\n");
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
            matvecs_csr(mtx_csr_ptr, vec, vec_res_sparse, num_mults);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; /* elapsed time */
    printf("  Sparse matrix %dx mult Serial time (s):   %9.6f\n", num_mults, elapsed_time);
    // print_matrix(mtx_p, rows, cols);
    // print_vector(vec, rows);
    // print_vector(vec_res_sparse, rows);
    printf("\nSparse matrix repeated multiplication PARALLEL...\n");
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
            matvecs_csr_parallel(mtx_csr_ptr, vec, vec_res_sparse_parallel, num_mults, thread_count);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; /* elapsed time */
    printf("  Sparse matrix %dx mult Parallel time (s): %9.6f\n", num_mults, elapsed_time);
    // print_vector(vec_res_sparse_parallel, rows);

    /* Compare the two resulting vectors */
    printf("\nComparing Serial & Parallel results...\n");
    nerrors = vectors_diffs(vec_res_sparse, vec_res_sparse_parallel, matrix_size);
    if (nerrors == 0) {
        printf("  Results match!\n");
    } else {
        printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
    }

    
    /* ------------------------------- Compare Dense vs CSR ---------------------------- */
    printf("\n================================================");
    printf("\nFINAL: Comparing Dense vs Sparse matrix (parallel) multiplication results...\n");
    nerrors = vectors_diffs(vec_res_parallel, vec_res_sparse_parallel, matrix_size);
    if (nerrors == 0) {
        printf("  Results match!\n");
    } else {
        printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
    }


    /* ------------------------------------ Cleanup ------------------------------------ */
    /* Free allocated memory */
    free(mtx_p[0]); // frees the contiguous data block
    free(mtx_p);
    free(vec);
    free(vec_res);
    free(vec_res_parallel);
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
   fprintf(stderr, "   sparsity: Percentage of zero-elements. Should be a float from 0 to 1.\n");
   fprintf(stderr, "   num_mults: Number of repeated multiplications. Should be non-negative.\n");
   fprintf(stderr, "   thread_count: Number of threads. Should be positive.\n");
   exit(0);
}  /* Usage */