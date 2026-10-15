#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

namespace {

// NumPy's NPY_ARRAY_ALIGNED bit. pybind11 intentionally exposes only layout
// policy bits, so keep this ABI flag local instead of requesting a conversion.
constexpr int kAlignedArrayFlag = 0x0100;

void validate_array(const py::array& array, const char* name) {
    if (!array.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + " must have native-endian float64 dtype");
    }
    if (array.ndim() != 3) {
        throw py::value_error(std::string(name) + " must be a 3D array");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if ((array.flags() & kAlignedArrayFlag) == 0) {
        throw py::value_error(std::string(name) + " must be aligned");
    }
}

void validate_same_shape(const py::array& reference, const py::array& candidate,
                         const char* name) {
    for (py::ssize_t axis = 0; axis < reference.ndim(); ++axis) {
        if (reference.shape(axis) != candidate.shape(axis)) {
            throw py::value_error(std::string(name) + " must have the same shape as a");
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

    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});
    const auto size = static_cast<std::size_t>(a.size());
    mac_kernel(static_cast<const double*>(a.data()),
               static_cast<const double*>(b.data()),
               static_cast<const double*>(c.data()),
               out.mutable_data(), size);
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(),
               py::arg("c").noconvert(), "Return the elementwise result a * b + c.");
}
