#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void check_input(const py::array& value, const std::string& name) {
    // equal, а не is: разные объекты dtype могут описывать один и тот же float64.
    // Для float64 с другим порядком байтов equal вернёт false.
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(name + ": expected native-endian float64, got " +
                             py::str(value.dtype()).cast<std::string>());
    }
    if (value.ndim() != 3) {
        throw py::value_error(name + ": expected 3 dimensions, got " +
                              std::to_string(value.ndim()));
    }
    if (!(value.flags() & py::array::c_style)) {
        throw py::value_error(name + ": array must be C-contiguous");
    }
    if (!value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(name + ": array must be aligned");
    }
}

bool same_shape(const py::array& x, const py::array& y) {
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (x.shape(axis) != y.shape(axis)) {
            return false;
        }
    }
    return true;
}

}  // namespace

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    check_input(a, "a");
    check_input(b, "b");
    check_input(c, "c");
    if (!same_shape(a, b) || !same_shape(a, c)) {
        throw py::value_error("a, b and c must have the same shape, broadcasting is not allowed");
    }

    // Новый массив со своим буфером, поэтому выход не делит память со входами.
    std::vector<py::ssize_t> shape = {a.shape(0), a.shape(1), a.shape(2)};
    py::array_t<double, py::array::c_style> out(shape);

    mac_kernel(static_cast<const double*>(a.data()),
               static_cast<const double*>(b.data()),
               static_cast<const double*>(c.data()),
               out.mutable_data(),
               static_cast<std::size_t>(a.size()));
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac,
               py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert(),
               "Return a * b + c elementwise for three C-contiguous float64 arrays of the same 3D shape");
}
