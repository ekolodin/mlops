#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void validate(const py::array& value) {
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error("expected native-endian float64 arrays");
    }
    if (value.ndim() != 3) {
        throw py::value_error("expected 3D arrays");
    }
    if (!(value.flags() & py::array::c_style) ||
        !value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error("expected C-contiguous, aligned arrays");
    }
}

py::array_t<double> mac(const py::array& a, const py::array& b, const py::array& c) {
    validate(a);
    validate(b);
    validate(c);
    for (int axis = 0; axis < 3; ++axis) {
        if (a.shape(axis) != b.shape(axis) || a.shape(axis) != c.shape(axis)) {
            throw py::value_error("array shapes must match");
        }
    }

    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});
    if (a.size() != 0) {
        mac_kernel(static_cast<const double*>(a.data()),
                   static_cast<const double*>(b.data()),
                   static_cast<const double*>(c.data()),
                   out.mutable_data(), static_cast<std::size_t>(a.size()));
    }
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(),
               py::arg("b").noconvert(), py::arg("c").noconvert());
}
