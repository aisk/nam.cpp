cmake_minimum_required(VERSION 3.16)
project(nam_ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GGML_DIR "" CACHE PATH "Path to a ggml checkout")
if(NOT GGML_DIR AND DEFINED ENV{GGML_DIR})
    set(GGML_DIR "$ENV{GGML_DIR}")
endif()
if(NOT EXISTS "${GGML_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "GGML_DIR must point to a ggml source directory. "
        "Pass -DGGML_DIR=/path/to/ggml or set the GGML_DIR environment variable.")
endif()

set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.2
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(argparse)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONC REQUIRED IMPORTED_TARGET json-c)

add_executable(nam
    src/inference.cpp
    src/main.cpp
    src/model.cpp
    src/wav.cpp)
target_link_libraries(nam PRIVATE argparse ggml PkgConfig::JSONC)
target_compile_options(nam PRIVATE -Wall -Wextra -Wpedantic)
