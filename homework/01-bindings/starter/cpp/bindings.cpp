#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;

static void check_input(const py::array& value, const char* name) {
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + ": expected native float64");
    }
    if (value.ndim() != 3) {
        throw py::value_error(std::string(name) + ": expected exactly 3 dimensions");
    }
    if (!(value.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + ": expected C-contiguous array");
    }
    if (!value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(std::string(name) + ": expected aligned array");
    }
}

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    check_input(a, "a");
    check_input(b, "b");
    check_input(c, "c");

    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (a.shape(axis) != b.shape(axis) || a.shape(axis) != c.shape(axis)) {
            throw py::value_error("a, b and c mut have the same shape");
        }
    }

    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});
    const auto size = static_cast<std::size_t>(a.size());
    if (size > 0) {
        mac_kernel(static_cast<const double*>(a.data()),
                   static_cast<const double*>(b.data()),
                   static_cast<const double*>(c.data()),
                   out.mutable_data(), size);
    }
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac,
               py::arg("a").noconvert(),
               py::arg("b").noconvert(),
               py::arg("c").noconvert());
}