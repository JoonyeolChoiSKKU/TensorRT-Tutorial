cmake_minimum_required(VERSION 3.10)
project(RunTRT CXX)

find_package(CUDA REQUIRED)

# TensorRT 헤더/라이브러리 경로 (Ubuntu apt 설치 기준)
include_directories("/usr/include/x86_64-linux-gnu")
link_directories("/usr/lib/x86_64-linux-gnu")

add_executable(run_trt main.cpp)
target_compile_features(run_trt PRIVATE cxx_std_11)

# 표준 라이브러리: nvinfer, CUDA
target_link_libraries(run_trt
    PRIVATE
    nvinfer
    ${CUDA_LIBRARIES}
)
