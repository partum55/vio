cmake_minimum_required(VERSION 3.16)
project(vio_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCL)

add_executable(vio
    main.cpp
    src/core/data_generator.cpp
    src/core/dataset.cpp
    src/core/dataset_streamer.cpp
    src/core/rerun_stream.cpp
    src/core/vio_runner.cpp
    src/frontend/vision_compute_backend.cpp
    src/imu/imu.cpp
    src/keypoints/gaussian_blur.cpp
    src/keypoints/sobel.cpp
    src/keypoints/shi_tomasi.cpp
    src/tracking/lk_tracker.cpp
)

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

    if(OpenCL_FOUND)
        target_compile_definitions(${target_name} PRIVATE VIO_HAVE_OPENCL=1)
        target_link_libraries(${target_name} PRIVATE OpenCL::OpenCL)
    endif()
endfunction()

configure_vio_target(vio)

add_executable(imu_tracking_app
    apps/imu_tracking_app.cpp
    src/core/dataset_streamer.cpp
    src/core/projection.cpp
    src/frontend/feature_extractor.cpp
    src/frontend/feature_tracker.cpp
    src/frontend/frame_pose_sync.cpp
    src/frontend/imu_tracking_pipeline.cpp
    src/frontend/pivot_frame.cpp
    src/frontend/vision_compute_backend.cpp
    src/frontend/visual_frontend.cpp
    src/imu/imu.cpp
    src/io/image_sequence_reader.cpp
    src/io/landmark_output_writer.cpp
    src/io/tracked_output_writer.cpp
    src/keypoint_extraction/gaussian_blur.cpp
    src/keypoint_extraction/shi_tomasi.cpp
    src/keypoint_extraction/sobel.cpp
    src/keypoints/gaussian_blur.cpp
    src/keypoints/shi_tomasi.cpp
    src/keypoints/sobel.cpp
    src/tracking/feature_refresh.cpp
    src/tracking/lk_tracker.cpp
    src/tracking/tracking_vis.cpp
    src/triangulation/triangulator.cpp
)

configure_vio_target(imu_tracking_app)
