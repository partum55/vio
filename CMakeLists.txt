cmake_minimum_required(VERSION 3.16)
project(vio_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
#find_package(OpenCL)

# =========================
# Common config
# =========================
function(configure_vio_target target_name)
    target_include_directories(${target_name} PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
    )

    target_compile_options(${target_name} PRIVATE
            -O3
            -march=native
            -ffast-math
    )

    target_link_libraries(${target_name} PRIVATE
            Eigen3::Eigen
            ${OpenCV_LIBS}
            TBB::tbb
            Threads::Threads
    )
endfunction()

# =========================
# MAIN VIO (optional)
# =========================
add_executable(vio
        main.cpp

        src/core/data_generator.cpp
        src/core/dataset.cpp
        src/core/dataset_streamer.cpp
        src/core/rerun_stream.cpp
        src/core/vio_runner.cpp

        src/frontend/vision_compute_backend.cpp

        src/imu/imu.cpp

        src/keypoint_extraction/gaussian_blur.cpp
        src/keypoint_extraction/sobel.cpp
        src/keypoint_extraction/shi_tomasi.cpp

        src/tracking/lk_tracker.cpp
)

configure_vio_target(vio)

# =========================
# IMU TRACKING APP (твій головний)
# =========================
add_executable(imu_tracking_app
        apps/imu_tracking_app.cpp

        # core
        src/core/dataset_streamer.cpp
        src/core/projection.cpp

        # frontend
        src/frontend/feature_extractor.cpp
        src/frontend/feature_tracker.cpp
        src/frontend/frame_pose_sync.cpp
        src/frontend/imu_tracking_pipeline.cpp
        src/frontend/pivot_frame.cpp
        src/frontend/vision_compute_backend.cpp
        src/frontend/visual_frontend.cpp

        # pipeline (людина 3)
        src/pipeline/vio_pipeline.cpp
        src/pipeline/pipeline_pivot.cpp

        # geometry (людина 3)
        src/geometry/geometry_backend.cpp
        src/geometry/triangulator.cpp
        src/geometry/pnp_solver.cpp
        src/geometry/landmark_map.cpp

        # imu
        src/imu/imu.cpp

        # io
        src/io/image_sequence_reader.cpp
        src/io/landmark_output_writer.cpp
        src/io/tracked_output_writer.cpp

        # keypoint extraction
        src/keypoint_extraction/gaussian_blur.cpp
        src/keypoint_extraction/sobel.cpp
        src/keypoint_extraction/shi_tomasi.cpp

        # tracking
        src/tracking/feature_refresh.cpp
        src/tracking/lk_tracker.cpp
        src/tracking/tracking_vis.cpp
)

configure_vio_target(imu_tracking_app)