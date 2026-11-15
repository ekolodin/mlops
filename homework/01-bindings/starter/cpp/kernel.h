#pragma once
#include <cstddef>

void mac_kernel(const double* a, const double* b, const double* c,
                double* out, std::size_t size);
