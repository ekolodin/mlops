#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;

static void require_float64(const py::array& value, const char* name) {
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + " must be float64");
    }
}

static void require_3d(const py::array& value, const char* name) {
    if (!(value.ndim() == 3)) {
        throw py::value_error(std::string(name) + " must be 3-dimensional");
    }
}
static void require_same_shape(const py::array& x, const py::array& y, const char* message) {
    for (py::ssize_t i = 0; i < 3; i++) {
        if (x.shape(i) != y.shape(i)) {
            throw py::value_error(message);
        }
    }
}

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    require_float64(a, "a");
    require_float64(b, "b");
    require_float64(c, "c");
    require_3d(a, "a");
    require_3d(b, "b");
    require_3d(c, "c");
    require_same_shape(a, b, "a and b must have the same shape");
    require_same_shape(a, c, "a and c must have the same shape");
    throw std::logic_error("TODO 2: implement array binding");
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
