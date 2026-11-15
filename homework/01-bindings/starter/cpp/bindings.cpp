#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

namespace {
void validate_array(const py::array& value, const char* name) {
    const std::string prefix = std::string(name) + ": ";
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(prefix + "expected native-endian float64");
    }
    if (value.ndim() != 3) {
        throw py::value_error(prefix + "expected exactly three dimensions");
    }
    if (!(value.flags() & py::array::c_style)) {
        throw py::value_error(prefix + "expected a C-contiguous array");
    }
    if (!value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(prefix + "expected an aligned array");
    }
}
}

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    validate_array(a, "a");
    validate_array(b, "b");
    validate_array(c, "c");
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (a.shape(axis) != b.shape(axis) || a.shape(axis) != c.shape(axis)) {
            throw py::value_error("a, b and c must have identical shapes");
        }
    }

    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});
    if (a.size() != 0) {
        mac_kernel(static_cast<const double*>(a.data()),
                   static_cast<const double*>(b.data()),
                   static_cast<const double*>(c.data()), out.mutable_data(),
                   static_cast<std::size_t>(a.size()));
    }
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(),
               py::arg("c").noconvert(),
               "Elementwise a * b + c for 3D float64 arrays.");
}
