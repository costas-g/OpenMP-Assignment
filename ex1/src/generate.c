#include <stdio.h>
#include <stdlib.h>

long long *generate_random_poly(size_t n, size_t max_coeff){
    long long *poly = malloc((size_t)(n+1) * sizeof(long long));
    if (!poly) {
        perror("malloc poly");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i <= n; i++){
        int coeff;
        do {
        coeff = (rand() % (2*max_coeff + 1)) - max_coeff; /* in range [-max_coeff, max_coeff] */
        } while (coeff == 0); /* coefficients should be non-zero integers */
        poly[i] = (long long) coeff;
    }

    return poly;
}