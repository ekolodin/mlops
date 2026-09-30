#include "kernel.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

namespace {

void require_float64_array(const py::array& value, const char* name) {
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + " must be a native-endian float64 ndarray");
    }
    if (value.ndim() != 3) {
        throw py::value_error(std::string(name) + " must have exactly 3 dimensions");
    }
    const bool c_contiguous = (value.flags() & py::array::c_style) != 0;
    const bool aligned = value.attr("flags").attr("aligned").cast<bool>();
    if (!c_contiguous || !aligned) {
        throw py::value_error(std::string(name) + " must be C-contiguous and aligned");
    }
}

void require_same_shape(const py::array& a, const py::array& b, const py::array& c) {
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (a.shape(axis) != b.shape(axis) || a.shape(axis) != c.shape(axis)) {
            throw py::value_error("inputs must have the same shape; broadcasting is not supported");
        }
    }
}

}  // namespace

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    require_float64_array(a, "a");
    require_float64_array(b, "b");
    require_float64_array(c, "c");
    require_same_shape(a, b, c);

    const py::ssize_t d0 = a.shape(0);
    const py::ssize_t d1 = a.shape(1);
    const py::ssize_t d2 = a.shape(2);
    py::array_t<double> out({d0, d1, d2});

    const auto n = static_cast<std::size_t>(d0) * static_cast<std::size_t>(d1) *
                   static_cast<std::size_t>(d2);
    if (n > 0) {
        mac_kernel(static_cast<const double*>(a.data()), static_cast<const double*>(b.data()),
                   static_cast<const double*>(c.data()), out.mutable_data(), n);
    }
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(),
               py::arg("c").noconvert());
}
