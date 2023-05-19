cmake_minimum_required(VERSION 3.18)


set(DCMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.2/bin/nvcc)
project(darknet LANGUAGES CXX CUDA)
include_directories(/usr/local/cuda/include)

include_directories(include)
add_subdirectory(src)
add_subdirectory(examples)
