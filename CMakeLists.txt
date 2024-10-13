cmake_minimum_required(VERSION 3.28)
project(KMeansWithCUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(KMeansWithCUDA main.cu
        StastManager.cu)

set_target_properties(KMeansWithCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
