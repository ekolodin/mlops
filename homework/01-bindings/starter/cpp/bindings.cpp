#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    if (a.ndim() != 3) {
        throw py::value_error("a must have exactly 3 dimensions");
    }
    if (b.ndim() != 3) {
        throw py::value_error("b must have exactly 3 dimensions");
    }
    if (c.ndim() != 3) {
        throw py::value_error("c must have exactly 3 dimensions");
    }

    if (!a.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error("a must have dtype float64 (native-endian)");
    }
    if (!b.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error("b must have dtype float64 (native-endian)");
    }
    if (!c.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error("c must have dtype float64 (native-endian)");
    }

    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1) || a.shape(2) != b.shape(2) ||
        a.shape(0) != c.shape(0) || a.shape(1) != c.shape(1) || a.shape(2) != c.shape(2)) {
        throw py::value_error("a, b, and c must have identical shapes");
    }

    if ((a.flags() & py::array::c_style) != py::array::c_style) {
        throw py::value_error("a must be C-contiguous");
    }
    if ((b.flags() & py::array::c_style) != py::array::c_style) {
        throw py::value_error("b must be C-contiguous");
    }
    if ((c.flags() & py::array::c_style) != py::array::c_style) {
        throw py::value_error("c must be C-contiguous");
    }

    if (!a.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error("a must be aligned");
    }
    if (!b.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error("b must be aligned");
    }
    if (!c.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error("c must be aligned");
    }

    const auto size = static_cast<std::size_t>(a.size());
    const auto* a_ptr = static_cast<const double*>(a.data());
    const auto* b_ptr = static_cast<const double*>(b.data());
    const auto* c_ptr = static_cast<const double*>(c.data());

    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});
    auto* out_ptr = static_cast<double*>(out.mutable_data());
    mac_kernel(a_ptr, b_ptr, c_ptr, out_ptr, size);
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
