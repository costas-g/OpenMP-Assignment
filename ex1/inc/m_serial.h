#ifndef _m_serial_h_
#define _m_serial_h_

#include <stddef.h> /* defines size_t */

long long *m_serial(const long long *A, size_t n, const long long *B, size_t m, double *time);

#endif