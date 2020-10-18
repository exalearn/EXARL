cmake_minimum_required(VERSION 3.9)
project (DotsAndBoxes)
 
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
 
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
# find_package(OpenMP)
# find_package(MPI)

# find_path(PYTHON NAME Python.h HINTS /people/suet688/.conda/envs/exalearn/include/python3.7m/)
# include_directories(${PYTHON})

# find_path(PYBIND11 NAME pybind11.h HINTS /people/suet688/.conda/envs/exalearn/lib/python3.7/site-packages/pybind11/include/pybind11)
# get_filename_component(PYBIND_INC ${PYBIND11} DIRECTORY)
# include_directories(${PYBIND_INC})

add_library(dotsAndBoxes SHARED dotsAndBoxes.cpp)
target_link_libraries(dotsAndBoxes rt)

add_executable(test test.cpp)
target_link_libraries(test dotsAndBoxes)

add_subdirectory(pybind11)
pybind11_add_module(DotAndBoxes dotsAndBoxes.cpp binding.cpp)
