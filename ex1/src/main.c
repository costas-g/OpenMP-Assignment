// #define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include "generate.h"
#include "m_parallel.h"
#include "m_serial.h"


int  main(int argc, char* argv[]) {
    if (argc != 3){
        fprintf(stderr, "Usage: %s <degree> <num_threads>\n", argv[0]);
        return 1;
    }

    int max = 9;
    size_t n; // example 10e6 
    size_t num_threads;
    double time, time_gen;
    struct timespec begin1,end1;
    //input values
    n = strtoul(argv[1], NULL, 10);
    num_threads = strtoul(argv[2], NULL, 10);

    printf("Generating Polynomials...\n");
    //Begining time for generate 
    clock_gettime(CLOCK_MONOTONIC, &begin1);
    //generate the polynomials. 
    const long long *A,*B;
    A = generate_random_poly(n, max);
    B = generate_random_poly(n, max);
    //complete the generation 
    //time calculation 
    clock_gettime(CLOCK_MONOTONIC, &end1);
    time_gen = (end1.tv_sec - begin1.tv_sec) + (end1.tv_nsec - begin1.tv_nsec) / 1e9;
    printf("    Generate Time (s): %9.6f\n", time_gen);  


 //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    long long *R_serial, *R_parallel;
    R_serial = calloc((size_t)2*n + 1, sizeof(long long)); // maybe size n + m + 1 | will see
    R_parallel = calloc((size_t)2*n + 1, sizeof(long long)); 

    //Calculate multiplication in serial 
    printf("\nSerial Multiplication...\n");
    R_serial = m_serial(A, n, B, n, &time);
    printf("    Serial Time (s):   %9.6f\n", time);
    double serial_time = time;

    //Calculate multiplication in parallel
    printf("\nParallel Multiplication...\n");
    R_parallel = m_parallel(A, n, B, n, num_threads, &time);
    printf("    Parallel Time (s): %9.6f\n", time);
    double parallel_time = time;

    //Calculate speedup
    printf("\nSpeedup: %.3f\n", serial_time/parallel_time);
    printf("\n");

    //````````````````````````````````````````````````````````````
    //time to compare them 
    for (size_t i = 0; i < 2*n + 1; i++){
        if (R_serial[i] != R_parallel[i]) {
                printf("Mismatch at i=%ld: serial=%lld, parallel=%lld\n", i,R_serial[i], R_parallel[i]);
                printf("ERROR\n");
                return 1;
        }
    }
    printf("Results match!\n");
    
    free(R_serial);
    free(R_parallel);
    return 0;
} /*main*/
