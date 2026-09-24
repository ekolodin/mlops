#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void validate(const py::array& value, const char* name) {
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(
            std::string(name) + " must have dtype float64 (native-endian)");
    }
    if (value.ndim() != 3) {
        throw py::value_error(std::string(name) + " must be 3-dimensional");
    }
    if (!(value.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (!value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(std::string(name) + " must be aligned");
    }
}

}  // namespace

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    validate(a, "a");
    validate(b, "b");
    validate(c, "c");

    if (a.shape(0) != b.shape(0) || a.shape(0) != c.shape(0) ||
        a.shape(1) != b.shape(1) || a.shape(1) != c.shape(1) ||
        a.shape(2) != b.shape(2) || a.shape(2) != c.shape(2)) {
        throw py::value_error("a, b, c must have identical shapes");
    }

    std::vector<py::ssize_t> shape = {a.shape(0), a.shape(1), a.shape(2)};
    py::array_t<double> out(shape);

    const std::size_t size = static_cast<std::size_t>(a.size());
    if (size > 0) {
        mac_kernel(static_cast<const double*>(a.data()),
                   static_cast<const double*>(b.data()),
                   static_cast<const double*>(c.data()),
                   out.mutable_data(), size);
    }
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise A * B + C on three 3D float64 NumPy arrays";
    module.def("mac", &mac,
               py::arg("a").noconvert(),
               py::arg("b").noconvert(),
               py::arg("c").noconvert());
}