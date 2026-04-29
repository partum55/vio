cmake_minimum_required(VERSION 3.16)
project(vio_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio video calib3d)
find_package(Threads REQUIRED)

add_executable(vio
        main.cpp

        src/core/projection.cpp

#        src/imu/imu_processor.cpp

        src/frontend/frame_pose_sync.cpp
        src/frontend/imu_tracking_pipeline.cpp
        src/frontend/pivot_frame.cpp

        src/io/image_sequence_reader.cpp
        src/io/landmark_output_writer.cpp
        src/io/tracked_output_writer.cpp

        src/keypoint_extraction/feature_refresh.cpp
        src/keypoint_extraction/gaussian_blur.cpp
        src/keypoint_extraction/shi_tomasi.cpp
        src/keypoint_extraction/sobel.cpp

        src/tracking/feature_refresh.cpp
        src/tracking/klt_tracker.cpp
        src/tracking/lk_tracker.cpp
        src/tracking/tracking_vis.cpp

        src/geometry/triangulator.cpp
        src/geometry/landmark_map.cpp
        src/geometry/pnp_solver.cpp
        src/geometry/geometry_backend.cpp

        src/pipeline/vio_pipeline.cpp
        src/pipeline/vio_app_pipeline.cpp
)

target_include_directories(vio PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vio PRIVATE
        Eigen3::Eigen
        ${OpenCV_LIBS}
        Threads::Threads
        stdc++fs
)