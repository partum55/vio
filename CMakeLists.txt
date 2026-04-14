cmake_minimum_required(VERSION 3.10)

project(shi_tomasi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(COMMON_SOURCES
    src/shi_tomasi.cpp
    src/gaussian_blur.cpp
    src/sobel.cpp
    src/feature_refresh.cpp
)

add_executable(shi_tomasi
    ${COMMON_SOURCES}
    src/compare_keypoint_extraction.cpp
)

add_executable(benchmark_shi_tomasi
    ${COMMON_SOURCES}
    src/benchmark_shi_tomasi.cpp
)

add_executable(benchmark_convolution_only
    ${COMMON_SOURCES}
    src/benchmark_convolution_only.cpp
)

target_include_directories(shi_tomasi PRIVATE include/)
target_include_directories(benchmark_shi_tomasi PRIVATE include/)
target_include_directories(benchmark_convolution_only PRIVATE include/)

target_link_libraries(shi_tomasi PRIVATE ${OpenCV_LIBS} Threads::Threads)
target_link_libraries(benchmark_shi_tomasi PRIVATE ${OpenCV_LIBS} Threads::Threads)
target_link_libraries(benchmark_convolution_only PRIVATE ${OpenCV_LIBS} Threads::Threads)

target_compile_options(shi_tomasi PRIVATE -O3 -march=native)
target_compile_options(benchmark_shi_tomasi PRIVATE -O3 -march=native)
target_compile_options(benchmark_convolution_only PRIVATE -O3 -march=native)

add_executable(benchmark_convolution_scalar
    ${COMMON_SOURCES}
    src/benchmark_convolution_only.cpp
)

target_include_directories(benchmark_convolution_scalar PRIVATE include/)
target_link_libraries(benchmark_convolution_scalar PRIVATE ${OpenCV_LIBS} Threads::Threads)
target_compile_options(benchmark_convolution_scalar PRIVATE -O3)