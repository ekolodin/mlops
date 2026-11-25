#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include "kernel.h"

namespace py = pybind11;

namespace {

std::string shape_str(const py::array& value) {
    std::string result = "(";
    for (py::ssize_t axis = 0; axis < value.ndim(); ++axis) {
        if (axis > 0) {
            result += ", ";
        }
        result += std::to_string(value.shape(axis));
    }
    return result + ")";
}

void check_input(const py::array& value, const std::string& name) {
    // equal() also compares byte order, so non-native float64 is rejected here.
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(name + " must have native-endian float64 dtype, got " +
                             py::str(value.dtype()).cast<std::string>());
    }
    if (value.ndim() != 3) {
        throw py::value_error(name + " must be 3-dimensional, got shape " + shape_str(value));
    }
    if (!(value.flags() & py::array::c_style)) {
        throw py::value_error(name + " must be C-contiguous");
    }
    if (!value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(name + " must be aligned");
    }
}

void check_same_shape(const py::array& expected, const py::array& value, const std::string& name) {
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (value.shape(axis) != expected.shape(axis)) {
            throw py::value_error(name + " has shape " + shape_str(value) + ", expected " +
                                  shape_str(expected));
        }
    }
}

}  // namespace

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    check_input(a, "a");
    check_input(b, "b");
    check_input(c, "c");
    check_same_shape(a, b, "b");
    check_same_shape(a, c, "c");

    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});
    const auto size = static_cast<std::size_t>(a.size());
    if (size > 0) {
        mac_kernel(static_cast<const double*>(a.data()), static_cast<const double*>(b.data()),
                   static_cast<const double*>(c.data()), out.mutable_data(), size);
    }
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(),
               py::arg("c").noconvert(),
               "Return a * b + c elementwise for three C-contiguous 3D float64 arrays");
}
