cmake_minimum_required(VERSION 3.16)
project(vio_imu_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vio_imu_test
    src/main.cpp
    src/imu.cpp
)

target_include_directories(vio_imu_test PRIVATE
    include
    /opt/homebrew/include/eigen3
)