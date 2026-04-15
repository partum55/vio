cmake_minimum_required(VERSION 3.16)
project(vio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(Threads REQUIRED)

add_executable(vio
    main.cpp
    src/core/data_generator.cpp
    src/core/dataset.cpp
    src/core/rerun_stream.cpp
    src/core/vio_runner.cpp
    src/imu/imu.cpp
    src/keypoints/gaussian_blur.cpp
    src/keypoints/sobel.cpp
    src/keypoints/shi_tomasi.cpp
    src/tracking/lk_tracker.cpp
)

target_include_directories(vio PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vio PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Threads::Threads
)

add_executable(imu_tracking_app
    apps/imu_tracking_app.cpp
    src/frontend/imu_tracking_pipeline.cpp
    src/frontend/frame_pose_sync.cpp
    src/io/image_sequence_reader.cpp
    src/io/tracked_output_writer.cpp
    src/imu/imu.cpp
    src/tracking/lk_tracker.cpp
    src/tracking/tracking_vis.cpp
    src/tracking/feature_refresh.cpp
    src/keypoints/shi_tomasi.cpp
    src/keypoints/gaussian_blur.cpp
    src/keypoints/sobel.cpp
)

target_include_directories(imu_tracking_app PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(imu_tracking_app PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Threads::Threads
)
