// #define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "generate.h"
#include "m_parallel.h"
#include "m_serial.h"

void Usage(char* prog_name);
// long long *generate_random_poly(size_t n, size_t max_coeff);

int  main(int argc, char* argv[]) {
    int n; /* degree of polynomials */
    int thread_count;

    /* Parse inputs and error check */
    if (argc != 3) Usage(argv[0]);

    n = strtol(argv[1], NULL, 10);
    if (n <= 0) Usage(argv[0]);

    thread_count = strtol(argv[2], NULL, 10);
    if (thread_count <= 0) Usage(argv[0]);

    /* Timing variables */
    struct timespec start, end;
    double time, time_gen;

    /* Generate the two polynomials */
    printf("Generating Polynomials...\n");
    size_t max_coeff = 9; /* maximum coefficient value (absolute value) */
    const long long *A, *B;
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    A = generate_random_poly((size_t) n, max_coeff);
    B = generate_random_poly((size_t) n, max_coeff);
    clock_gettime(CLOCK_MONOTONIC, &end); /* end time */

    /* elapsed time */
    time_gen = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; 
    printf("  Generate Time (s): %9.6f\n", time_gen);  


    /* Polynomial Multiplication */
    long long *R_serial, *R_parallel;
    R_serial = calloc((size_t)2*n + 1, sizeof(long long)); /* degree of result is n+m, so size is n+m+1 */
    R_parallel = calloc((size_t)2*n + 1, sizeof(long long)); 
    if (!R_serial) {
        perror("calloc R_serial");
        exit(EXIT_FAILURE);
    }
    if (!R_parallel) {
        perror("calloc R_parallel");
        exit(EXIT_FAILURE);
    }

    /* Serial Poly Multiplication */ 
    printf("\nSerial Multiplication...\n");
    R_serial = m_serial(A, n, B, n, &time);
    printf("  Serial Time (s):   %9.6f\n", time);
    double serial_time = time;

    /* Parallel Poly Multiplication */ 
    printf("\nParallel Multiplication...\n");
    R_parallel = m_parallel(A, n, B, n, thread_count, &time);
    printf("  Parallel Time (s): %9.6f\n", time);
    double parallel_time = time;

    /* Speedup calculation */ 
    printf("\nSpeedup: %.3f\n", serial_time/parallel_time);
    printf("\n");

    /* Confirm parallel result correctness */
    for (size_t i = 0; i < (size_t) 2*n + 1; i++){
        if (R_serial[i] != R_parallel[i]) {
            printf("Mismatch at i=%ld: serial=%lld, parallel=%lld\n", i, R_serial[i], R_parallel[i]);
            printf("ERROR\n");
            return 1;
        }
    }
    printf("Results match!\n");
    
    /* Free allocated memory */
    free(R_serial);
    free(R_parallel);

    return 0;
} /* main */

/*--------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Print a message indicating how program should be started
 *            and terminate.
 */
void Usage(char *prog_name) {
   fprintf(stderr, "Usage: %s <degree> <thread_count>\n", prog_name);
   fprintf(stderr, "   degree should be positive\n");
   fprintf(stderr, "   thread_count should be positive\n");
   exit(0);
}  /* Usage */