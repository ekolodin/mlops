#include "kernel.h"
#include <stdexcept>

void mac_kernel(const double* a, const double* b, const double* c,
                double* out, std::size_t size) {
    // TODO 1: compute the elementwise operation in C++, without calling NumPy.
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
}
