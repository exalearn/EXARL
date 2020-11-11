cmake_minimum_required(VERSION 3.9)
project (DotsAndBoxes)
 
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug Build")
    set(CMAKE_C_FLAGS "-g -Werror -O0")
    set(CMAKE_CXX_FLAGS "-g -Werror -O0")
else()
    message("Release Build")
    set(CMAKE_C_FLAGS "-g -O3 -march=native")
    set(CMAKE_CXX_FLAGS "-g -O3 -march=native")
endif()

#Find Required Libs-------------------------->

find_library(RT_LIB rt)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(dotsandboxes src/binding.cpp src/dotsAndBoxes.cpp)
target_link_libraries(dotsandboxes PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)

add_executable(test src/test.cpp src/dotsAndBoxes.cpp)
target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)

# add_executable(mcst src/mcst.cpp)