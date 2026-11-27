#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <string>

namespace py = pybind11;

namespace {

void validate_array(const py::array& value, const char* name) {
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(
            std::string(name) + " must have native-endian float64 dtype"
        );
    }

    if (value.ndim() != 3) {
        throw py::value_error(
            std::string(name) + " must be a 3D array"
        );
    }

    if ((value.flags() & py::array::c_style) == 0) {
        throw py::value_error(
            std::string(name) + " must be C-contiguous"
        );
    }

    const bool aligned =
        value.attr("flags").attr("aligned").cast<bool>();

    if (!aligned) {
        throw py::value_error(
            std::string(name) + " must be aligned"
        );
    }
}

void validate_same_shape(
    const py::array& first,
    const py::array& second,
    const char* second_name
) {
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (first.shape(axis) != second.shape(axis)) {
            throw py::value_error(
                std::string(second_name) +
                " must have the same shape as a"
            );
        }
    }
}

}  // namespace

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    validate_array(a, "a");
    validate_array(b, "b");
    validate_array(c, "c");

    validate_same_shape(a, b, "b");
    validate_same_shape(a, c, "c");

    py::array_t<double> out({
        a.shape(0),
        a.shape(1),
        a.shape(2),
    });

    const auto* a_data = static_cast<const double*>(a.data());
    const auto* b_data = static_cast<const double*>(b.data());
    const auto* c_data = static_cast<const double*>(c.data());
    auto* out_data = out.mutable_data();

    mac_kernel(
        a_data,
        b_data,
        c_data,
        out_data,
        static_cast<std::size_t>(a.size())
    );

    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def(
        "mac",
        &mac,
        py::arg("a").noconvert(),
        py::arg("b").noconvert(),
        py::arg("c").noconvert(),
        "Compute the elementwise operation a * b + c."
    );
}
