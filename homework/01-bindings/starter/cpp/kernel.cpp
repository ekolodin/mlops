#include "kernel.h"

void mac_kernel(const double* a, const double* b, const double* c,
                double* out, std::size_t size) {
    // При size == 0 цикл не выполняется, указатели не разыменовываются.
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
}
