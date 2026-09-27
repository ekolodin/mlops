cmake_minimum_required(VERSION 3.26)
project(tensor_ops LANGUAGES CXX)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core cpp/bindings.cpp cpp/kernel.cpp)
target_compile_features(_core PRIVATE cxx_std_17)
install(TARGETS _core LIBRARY DESTINATION tensor_ops RUNTIME DESTINATION tensor_ops)

if (MINGW)
    get_filename_component(MINGW_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    foreach(runtime_dll IN ITEMS
        "libgcc_s_seh-1.dll"
        "libstdc++-6.dll"
        "libwinpthread-1.dll")
        set(runtime_dll_path "${MINGW_BIN_DIR}/${runtime_dll}")
        if (EXISTS "${runtime_dll_path}")
            install(FILES "${runtime_dll_path}" DESTINATION tensor_ops)
        endif()
    endforeach()
endif()
