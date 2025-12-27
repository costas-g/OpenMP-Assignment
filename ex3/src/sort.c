#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "merge_serial.h"
#include "merge_parallel.h"
#include "gen_rand_int_array.h"

void Usage(char* prog_name);

int  main(int argc, char* argv[]) {
    long long size;         /* size of integer array */
    int parallel_mode = 0;  /* default serial mode */
    int thread_count;

    /* Parse inputs and error check */
    if (argc < 3) Usage(argv[0]);

    size = strtoll(argv[1], NULL, 10);
    if (size <= 0) Usage(argv[0]);

    char sorp = argv[2][0]; /* serial or parallel execution by user input */
    if (sorp != 's' && sorp != 'S' && sorp != 'p' && sorp != 'P') Usage(argv[0]);
    if (sorp == 'p' || sorp == 'P') {
        parallel_mode = 1;
        if (argc < 4) Usage(argv[0]);
        thread_count = strtol(argv[3], NULL, 10);
        if (thread_count <= 0) Usage(argv[0]);
        printf("Selected Parallel Mergesort with %d threads\n", thread_count);
    }
    else
        printf("Selected Serial Mergesort\n");

    /* Timing variables */
    struct timespec start, end;
    double elapsed_time, time_gen;

    /* -------------------- Generate the array of integers ---------------------- */
    printf("Generating Array of integers...\n");
    int *A; /* pointer to the array of integers */
    srand((unsigned) time(NULL));
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    A = gen_rand_int_array(size);
    clock_gettime(CLOCK_MONOTONIC, &end); /* end time */

    /* elapsed time */
    time_gen = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; 
    printf("  Generate Time (s): %9.6f\n", time_gen);  


    /* -------------------------------- Sorting --------------------------------- */

    /* Allocate temp array to be used by the algorithms */
    int *tmp = malloc( size * sizeof(int));

    if(!parallel_mode)
    {
        /* Serial MergeSort */ 
        printf("\nSerial Mergesort...\n");
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
        mergesort(A, tmp, 0, size-1);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
        elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; 
        printf("  Serial Time (s):   %9.6f\n", elapsed_time);
    }
    else
    {
        /* Parallel MergeSort */ 
        printf("\nParallel Mergesort...\n");
        clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
        begin_parallel_mergesort(A, tmp, 0, size-1, (size_t) thread_count);
        clock_gettime(CLOCK_MONOTONIC, &end); /* end time */
        elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; 
        printf("  Parallel Time (s):   %9.6f\n", elapsed_time);
    }

    /* ---------------------------- Confirm sorting correctness ---------------------------- */
    int correct_sorting = 1;
    for (long long i = 0; i < size-1 ; i++){
        if (A[i] > A[i+1]) {
            printf("  Mistake at i=%lld: A[%lld] = %d > %d = A[%lld]\n", i, i, A[i], A[i+1], i+1);
            correct_sorting = 0;
            break;
        }
    }

    if (correct_sorting) {
        printf("\nCorrect sorting!\n");
        /* Print up to 20 first elements just to check */
        for(int i = 0; i < (size>20 ? 20 : size); i++)
            printf("%d, ", A[i]);
        puts("");
    } else {
        printf("\nERROR: Incorrect sorting!\n");
    }

    /* Free allocated memory */
    free(tmp);

    return 0;
} /* main */

/*--------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Print a message indicating how program should be started
 *            and terminate.
 */
void Usage(char *prog_name) {
   fprintf(stderr, "Usage: %s <array_size> <serial_or_parallel> [<thread_count>]\n", prog_name);
   fprintf(stderr, "   array_size should be positive\n");
   fprintf(stderr, "   serial_or_parallel should be either 's' or 'p'\n");
   fprintf(stderr, "   thread_count should be positive (must be specified if parallel is selected)\n");
   exit(0);
}  /* Usage */