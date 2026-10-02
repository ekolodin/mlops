#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    if (!a.dtype().equal(py::dtype::of<double>()) ||
        !b.dtype().equal(py::dtype::of<double>()) ||
        !c.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error("All inputs must have native-endian float64 dtype.");
    }
    if (a.ndim() != 3 || b.ndim() != 3 || c.ndim() != 3) {
        throw std::invalid_argument("All input arrays must be 3D.");
    }
    if (a.shape(0) != b.shape(0) || a.shape(0) != c.shape(0) ||
        a.shape(1) != b.shape(1) || a.shape(1) != c.shape(1) ||
        a.shape(2) != b.shape(2) || a.shape(2) != c.shape(2)) {
        throw std::invalid_argument("All input arrays must have the same shape.");
    }
    if ((a.flags() & py::array::c_style) == 0 || (b.flags() & py::array::c_style) == 0 || (c.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument("All input arrays must be contiguous.");
    }
    if (
        !a.attr("flags").attr("aligned").cast<bool>() || 
        !b.attr("flags").attr("aligned").cast<bool>() || 
        !c.attr("flags").attr("aligned").cast<bool>()
    ) {
        throw std::invalid_argument("All input arrays must be aligned.");
    }
    const double* a_ptr = static_cast<const double*>(a.data());
    const double* b_ptr = static_cast<const double*>(b.data());
    const double* c_ptr = static_cast<const double*>(c.data());
    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});
    double* out_ptr = static_cast<double*>(out.mutable_data());
    const auto size = static_cast<std::size_t>(a.size());
    mac_kernel(a_ptr, b_ptr, c_ptr, out_ptr, size);
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
