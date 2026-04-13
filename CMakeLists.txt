cmake_minimum_required(VERSION 3.15)
project(vio_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(SOURCES
        src/imu/imu.cpp

        src/tracking/lk_tracker.cpp
        src/tracking/tracking_vis.cpp
        src/tracking/feature_refresh.cpp

        src/frontend/frame_pose_sync.cpp
        src/frontend/imu_tracking_pipeline.cpp

        src/io/image_sequence_reader.cpp
        src/io/tracked_output_writer.cpp
        src/io/landmark_output_writer.cpp

        src/triangulation/triangulator.cpp

        src/core/projection.cpp
)

add_executable(imu_tracking_app
        apps/imu_tracking_app.cpp
        ${SOURCES}
)

target_include_directories(imu_tracking_app
        PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(imu_tracking_app
        PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)