#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    // TODO 2: validate dtype, ndim, matching shapes, contiguous/aligned buffers.
    // Allocate an independent output, call mac_kernel, and return the array.
    // The Python contract and exception types are specified in ../README.md.
    throw std::logic_error("TODO 2: implement array binding");
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
