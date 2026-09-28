#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

void check_array(const py::array& value) {
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error("expected native-endian float64");
    }
    if (value.ndim() != 3) {
        throw py::value_error("expected a 3D array");
    }
    if (!(value.flags() & py::array::c_style) ||
        !value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error("expected a C-contiguous, aligned array");
    }
}

py::array_t<double> mac(const py::array& a, const py::array& b, const py::array& c) {
    check_array(a);
    check_array(b);
    check_array(c);

    for (int axis = 0; axis < 3; ++axis) {
        if (a.shape(axis) != b.shape(axis) || a.shape(axis) != c.shape(axis)) {
            throw py::value_error("arrays must have the same shape");
        }
    }

    std::vector<py::ssize_t> shape{a.shape(0), a.shape(1), a.shape(2)};
    py::array_t<double> out(shape);
    mac_kernel(static_cast<const double*>(a.data()),
               static_cast<const double*>(b.data()),
               static_cast<const double*>(c.data()),
               out.mutable_data(), static_cast<std::size_t>(a.size()));
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(),
               py::arg("c").noconvert());
}
