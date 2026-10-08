#include "kernel.h"

void mac_kernel(const double* a, const double* b, const double* c,
                double* out, std::size_t size) {
    // In particular, do not touch pointers for a valid empty NumPy array.
    for (std::size_t index = 0; index < size; ++index) {
        out[index] = a[index] * b[index] + c[index];
    }
}
