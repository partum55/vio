cmake_minimum_required(VERSION 3.10)

project(shi_tomasi)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(shi_tomasi
        src/shi_tomasi.cpp
        src/compare_keypoint_extraction.cpp
)

add_executable(sobel
        src/sobel.cpp
        src/gaussian_blur.cpp
)

target_include_directories(shi_tomasi PRIVATE include/)
target_include_directories(sobel PRIVATE include/)

target_link_libraries(shi_tomasi ${OpenCV_LIBS})
target_link_libraries(sobel ${OpenCV_LIBS})