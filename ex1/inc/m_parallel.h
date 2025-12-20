#ifndef _m_parallel_h_
#define _m_parallel_h_

#include <stddef.h> /* defines size_t */

long long *m_parallel(const long long *A, size_t n, const long long *B, size_t m, size_t num_threads, double *time);

#endif