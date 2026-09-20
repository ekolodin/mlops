#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <vector>

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
static void require_c_contiguous(const py::array& value, const char* name) {
    if (!(value.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
}

static void require_aligned(const py::array& value, const char* name) {
    if (!value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(std::string(name) + " must be aligned");
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
    require_c_contiguous(a, "a");
    require_c_contiguous(b, "b");
    require_c_contiguous(c, "c");
    require_aligned(a, "a");
    require_aligned(b, "b");
    require_aligned(c, "c");

    std::vector<py::ssize_t> shape(3);
    for (py::ssize_t i = 0; i < 3; i++) {
        shape[i] = a.shape(i);
    }
    py::array_t<double> out(shape);
    mac_kernel(
        static_cast<const double*>(a.data()),
        static_cast<const double*>(b.data()),
        static_cast<const double*>(c.data()),
        out.mutable_data(),
        static_cast<std::size_t>(a.size())
    );
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
