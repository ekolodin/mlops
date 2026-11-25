#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    auto validate_array = [](const py::array& arr) {
        if (!arr.dtype().equal(py::dtype::of<double>())) {
            throw py::type_error("Expected float64 array");
        }
        if (arr.ndim() != 3) {
            throw py::value_error("Expected 3 dim array");
        }
        if (!(arr.flags() & py::array::c_style)) {
            throw py::value_error("Expected C-contiguous array");
        }
        if (!arr.attr("flags").attr("aligned").cast<bool>()) {
            throw py::value_error("Expected aligned array");
        }
    };
    validate_array(a);
    validate_array(b);
    validate_array(c);
    for (int axis = 0; axis < 3; axis++){
        if (a.shape(axis) != b.shape(axis) || a.shape(axis) != c.shape(axis)){
            throw py::value_error("Shapes don't match");
        }
    }
    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});
    mac_kernel(
        static_cast<const double*>(a.data()),
        static_cast<const double*>(b.data()),
        static_cast<const double*>(c.data()),
        static_cast<double*>(out.mutable_data()),
        static_cast<std::size_t>(a.size())
    );
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
