#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;
void validate_array(const py::array& value){
    // native-endian float64
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error("expected native-endian float64 NumPy array");
    }
    // 3D arrays
    if (value.ndim() != 3) {
        throw py::value_error("expected a 3D array");
    }
    //C-order
    if ((value.flags() & py::array::c_style) == 0) {
        throw py::value_error("expected a C-contiguous array");
    }
    //correct memory-work
    const bool aligned =
        value.attr("flags").attr("aligned").cast<bool>();

    if (!aligned) {
        throw py::value_error("expected an aligned array");
    }


}
py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    validate_array(a);
    validate_array(b);
    validate_array(c);
    //throw std::logic_error("TODO 2: implement array binding");
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (a.shape(axis) != b.shape(axis) ||
            a.shape(axis) != c.shape(axis)) {
            throw py::value_error("all input arrays must have the same shape");
        }
    }

    py::array_t<double> out({
        a.shape(0),
        a.shape(1),
        a.shape(2)
    });
    const std::size_t size =
        static_cast<std::size_t>(a.size());

    if (size == 0) {
        return out;
    }
    const double* a_ptr =
        static_cast<const double*>(a.data());

    const double* b_ptr =
        static_cast<const double*>(b.data());

    const double* c_ptr =
        static_cast<const double*>(c.data());

    double* out_ptr = out.mutable_data();
    //////////////////////////////////////////

    mac_kernel(a_ptr,b_ptr,c_ptr,out_ptr,size);

    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    // TODO 2: expose mac(a, b, c). Do not allow implicit argument conversions.
    module.def(
        "mac",
        &mac,
        py::arg("a").noconvert(), //non auto convert in numpy array
        py::arg("b").noconvert(),
        py::arg("c").noconvert()
    );
}
