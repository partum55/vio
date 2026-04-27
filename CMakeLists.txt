cmake_minimum_required(VERSION 3.16)
project(vio_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

add_executable(vio
    main.cpp
    src/core/data_generator.cpp
    src/core/dataset.cpp
    src/core/dataset_streamer.cpp
    src/core/rerun_stream.cpp
    src/core/vio_runner.cpp
    src/imu/imu.cpp
    src/keypoints/gaussian_blur.cpp
    src/keypoints/sobel.cpp
    src/keypoints/shi_tomasi.cpp
    src/tracking/klt_tracker.cpp
)

target_include_directories(vio PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(vio PRIVATE
        -O3
        -march=native
        -ffast-math
)

target_link_libraries(vio PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    TBB::tbb
    Threads::Threads
)
