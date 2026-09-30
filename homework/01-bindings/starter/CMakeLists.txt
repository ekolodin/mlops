cmake_minimum_required(VERSION 3.26)
project(tensor_ops LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module
)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
    cpp/bindings.cpp
    cpp/kernel.cpp
)

set_target_properties(_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

install(TARGETS _core
    LIBRARY DESTINATION tensor_ops
    RUNTIME DESTINATION tensor_ops
)