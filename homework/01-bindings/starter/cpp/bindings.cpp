#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <vector>

#include "kernel.h"

namespace py = pybind11;

namespace {

void validate_array(const py::array& value, const char* name) {
    const std::string prefix = std::string(name) + ": ";

    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(
            prefix + "expected native-endian float64"
        );
    }

    if (value.ndim() != 3) {
        throw py::value_error(
            prefix + "expected exactly 3 dimensions"
        );
    }

    if ((value.flags() & py::array::c_style) == 0) {
        throw py::value_error(
            prefix + "expected C-contiguous array"
        );
    }

    if (!value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(
            prefix + "expected aligned array"
        );
    }
}

py::array_t<double> mac(const py::array& a,
                       const py::array& b,
                       const py::array& c) {
    validate_array(a, "a");
    validate_array(b, "b");
    validate_array(c, "c");

    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (a.shape(axis) != b.shape(axis) ||
            a.shape(axis) != c.shape(axis)) {
            throw py::value_error(
                "a, b and c must have identical shapes"
            );
        }
    }

    const std::vector<py::ssize_t> shape = {
        a.shape(0), a.shape(1), a.shape(2)
    };

    py::array_t<double> out(shape);
    const auto size = static_cast<std::size_t>(a.size());

    if (size == 0) {
        return out;
    }

    mac_kernel(
        static_cast<const double*>(a.data()),
        static_cast<const double*>(b.data()),
        static_cast<const double*>(c.data()),
        out.mutable_data(),
        size
    );

    return out;
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise A * B + C for 3D NumPy arrays";

    module.def(
        "mac",
        &mac,
        py::arg("a").noconvert(),
        py::arg("b").noconvert(),
        py::arg("c").noconvert(),
        "Return a new array containing a * b + c."
    );
}