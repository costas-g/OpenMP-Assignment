#include <stdio.h>
#include <stdlib.h>

//here we generate a polynomial that we will use it in main 

long long  *generate_random_poly(int n, int max){
    if(n < 0 || max < 1) return NULL;

    long long *poly = malloc((size_t)(n+1) * sizeof(long long));
    if (!poly) return NULL;

    for (int i = 0; i <= n; ++i){
        int coeff;
        do {
        coeff = (rand() % (2*max + 1)) - max; // [-max, max]
        } while (coeff == 0);
        poly[i] = (long long) coeff;
    }

    return poly;
}