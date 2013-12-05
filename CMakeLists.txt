cmake_minimum_required(VERSION 2.6)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_INSTALL_PREFIX /opt)
project(opencl_app)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(OPENCL)
include_directories(${OPENCL_INCLUDE_DIR})

add_definitions(-std=c++0x -g -fpermissive -o0 2>&1)

set(CMAKE_CXX_FLAGS  -DCL_USE_DEPRECATED_OPENCL_1_1_APIS)

FILE(GLOB cpp_files "source/*.cpp")
FILE(GLOB hpp_files "source/*.hpp")

add_executable(opencl_app ${cpp_files} ${hpp_files})
target_link_libraries(opencl_app ${OPENCL_LIBRARIES})

