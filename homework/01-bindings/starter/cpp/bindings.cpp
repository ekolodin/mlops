#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

void validate_inputs(
    const py::array& a,
    const py::array& b,
    const py::array& c
) {
    const py::array* arrays[] = {&a, &b, &c};

    for (const py::array* value : arrays) {
        if (!value->dtype().equal(py::dtype::of<double>())) {
            throw py::type_error(
                "a, b, and c must have native-endian dtype"
            );
        }

        if (value->ndim() != 3) {
            throw py::value_error(
                "a, b, and c must be 3-dimensional arrays"
            );
        }

        if ((value->flags() & py::array::c_style) == 0) {
            throw py::value_error(
                "a, b, and c must be C-contiguous"
            );
        }

        if (!value->attr("flags").attr("aligned").cast<bool>()) {
            throw py::value_error(
                "a, b, and c must be aligned"
            );
        }
    }

    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (a.shape(axis) != b.shape(axis) ||
            a.shape(axis) != c.shape(axis)) {
            throw py::value_error(
                "a, b, and c must have identical shapes"
            );
        }
    }
}

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    // TODO 2: validate dtype, ndim, matching shapes, contiguous/aligned buffers.
    // Allocate an independent output, call mac_kernel, and return the array.
    // The Python contract and exception types are specified in ../README.md.

    validate_inputs(a, b, c);

    py::array_t<double> out({
        a.shape(0),
        a.shape(1),
        a.shape(2),
    });

    const auto count = static_cast<std::size_t>(a.size());

    if (count != 0) {
        mac_kernel(
            static_cast<const double*>(a.data()),
            static_cast<const double*>(b.data()),
            static_cast<const double*>(c.data()),
            static_cast<double*>(out.mutable_data()),
            count
        );
    }

    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    // TODO 2: expose mac(a, b, c). Do not allow implicit argument conversions.

    module.def(
        "mac",
        &mac,
        py::arg("a").noconvert(),
        py::arg("b").noconvert(),
        py::arg("c").noconvert(),
        "Compute elementwise a * b + c for three compatible 3D double arrays."
    );
}
