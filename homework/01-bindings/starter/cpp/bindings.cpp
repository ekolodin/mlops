#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

static void check_input(const py::array& x, const char* name) {
    if (!x.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + ": expected native float64 array");
    }
    if (x.ndim() != 3) {
        throw py::value_error(std::string(name) + ": expected 3D array, got ndim=" +
                              std::to_string(x.ndim()));
    }
    if (!(x.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + ": array must be C-contiguous");
    }
    if (!x.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(std::string(name) + ": array must be aligned");
    }
}

static bool same_shape(const py::array& x, const py::array& y) {
    for (py::ssize_t i = 0; i < 3; ++i) {
        if (x.shape(i) != y.shape(i)) {
            return false;
        }
    }
    return true;
}

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    check_input(a, "a");
    check_input(b, "b");
    check_input(c, "c");
    if (!same_shape(a, b) || !same_shape(a, c)) {
        throw py::value_error("a, b and c must have the same shape");
    }

    std::vector<py::ssize_t> shape = {a.shape(0), a.shape(1), a.shape(2)};
    py::array_t<double> out(shape);

    auto n = static_cast<std::size_t>(a.size());
    if (n > 0) {
        mac_kernel(static_cast<const double*>(a.data()),
                   static_cast<const double*>(b.data()),
                   static_cast<const double*>(c.data()),
                   out.mutable_data(), n);
    }
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac,
               py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert(),
               "Return a * b + c for three C-contiguous 3D float64 arrays of the same shape");
}
