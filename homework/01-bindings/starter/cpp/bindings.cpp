#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    if (!a.dtype().equal(py::dtype::of<double>()) || !b.dtype().equal(py::dtype::of<double>()) || !c.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error("All input arrays must be of type float64.");
    }
    if (a.ndim() != 3 || b.ndim() != 3 || c.ndim() != 3) {
        throw py::value_error("All input arrays must be 3-dimensional.");
    }
    if ((a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1) || a.shape(2) != b.shape(2)) || (a.shape(0) != c.shape(0) || a.shape(1) != c.shape(1) || a.shape(2) != c.shape(2))) {
        throw py::value_error("All input arrays must have the same shape."); 
    }
    if (!(a.flags() & py::array::c_style) || !(b.flags() & py::array::c_style) || !(c.flags() & py::array::c_style)) {
        throw py::value_error("All input arrays must be C-contiguous.");
    }
    if (!a.attr("flags").attr("aligned").cast<bool>() || !b.attr("flags").attr("aligned").cast<bool>() || !c.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error("All input arrays must be aligned.");
    }
    py::array out = py::array(py::dtype::of<double>(), {a.shape(0), a.shape(1), a.shape(2)});
    mac_kernel(static_cast<const double*>(a.data()),
               static_cast<const double*>(b.data()),
               static_cast<const double*>(c.data()),
               static_cast<double*>(out.mutable_data()),
               a.size());
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
