cmake_minimum_required(VERSION 3.10)

project(shi_tomasi)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(SOURCES
    src/shi_tomasi.cpp
    src/gaussian_blur.cpp
    src/sobel.cpp
)

add_executable(shi_tomasi
    ${SOURCES}
    src/compare_keypoint_extraction.cpp
)

add_executable(benchmark_shi_tomasi
    ${SOURCES}
    src/benchmark_shi_tomasi.cpp
)

target_include_directories(shi_tomasi PRIVATE include/)
target_include_directories(benchmark_shi_tomasi PRIVATE include/)

target_link_libraries(shi_tomasi ${OpenCV_LIBS})
target_link_libraries(benchmark_shi_tomasi ${OpenCV_LIBS})

target_compile_options(shi_tomasi PRIVATE -O3 -march=native)
target_compile_options(benchmark_shi_tomasi PRIVATE -O3 -march=native)