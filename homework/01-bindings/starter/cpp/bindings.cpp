#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    if (!a.dtype().equal(py::dtype::of<double>()) || !b.dtype().equal(py::dtype::of<double>()) || !c.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error("Expected float64 arrays");
    }

    if (a.ndim() != 3 || b.ndim() != 3 || c.ndim() != 3) {
        throw py::value_error("Expected 3D arrays");
    }

    for (int axis = 0; axis < 3; ++axis) {
        if (a.shape(axis) != b.shape(axis) || a.shape(axis) != c.shape(axis)) {
            throw py::value_error("Shapes must match");
        }
    }

    for (const auto* arr : {&a, &b, &c}) {
        if (!(arr->flags() & py::array::c_style)) {
            throw py::value_error("Array must be C-contiguous");
        }

        if (!arr->attr("flags").attr("aligned").cast<bool>()) {
            throw py::value_error("Array must be aligned");
        }
    }

    py::array_t<double> result({a.shape(0), a.shape(1), a.shape(2)});
    mac_kernel(static_cast<const double*>(a.data()), static_cast<const double*>(b.data()), static_cast<const double*>(c.data()), result.mutable_data(), static_cast<std::size_t>(a.size()));
    return result;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
