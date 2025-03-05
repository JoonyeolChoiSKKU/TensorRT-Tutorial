cmake_minimum_required(VERSION 3.10)
project(TensorRT_ONNX_Inference CXX)

find_package(CUDA REQUIRED)

# TensorRT 8.5+/10.x 설치되어 있다고 가정
# /usr/include/x86_64-linux-gnu, /usr/lib/x86_64-linux-gnu 등
include_directories("/usr/include/x86_64-linux-gnu")
link_directories("/usr/lib/x86_64-linux-gnu")

add_executable(run_trt main.cpp)

target_compile_features(run_trt PRIVATE cxx_std_11)

# nvinfer      -> /usr/lib/x86_64-linux-gnu/libnvinfer.so
# nvonnxparser -> /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
target_link_libraries(run_trt
    PRIVATE
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
)
