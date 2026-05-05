cmake_minimum_required(VERSION 3.16)
project(vio_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ENABLE_GPU_ACCELERATION "AUTO" CACHE STRING "Enable OpenCL acceleration: AUTO, ON, or OFF")
set_property(CACHE ENABLE_GPU_ACCELERATION PROPERTY STRINGS AUTO ON OFF)
option(ENABLE_RERUN_VISUALIZATION "Enable Rerun visualization" ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video videoio calib3d)
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

string(TOUPPER "${ENABLE_GPU_ACCELERATION}" VIO_GPU_ACCELERATION_MODE)
if(VIO_GPU_ACCELERATION_MODE STREQUAL "TRUE" OR VIO_GPU_ACCELERATION_MODE STREQUAL "1")
    set(VIO_GPU_ACCELERATION_MODE "ON")
elseif(VIO_GPU_ACCELERATION_MODE STREQUAL "FALSE" OR VIO_GPU_ACCELERATION_MODE STREQUAL "0")
    set(VIO_GPU_ACCELERATION_MODE "OFF")
endif()

if(NOT VIO_GPU_ACCELERATION_MODE STREQUAL "AUTO" AND
   NOT VIO_GPU_ACCELERATION_MODE STREQUAL "ON" AND
   NOT VIO_GPU_ACCELERATION_MODE STREQUAL "OFF")
    message(WARNING "Unknown ENABLE_GPU_ACCELERATION='${ENABLE_GPU_ACCELERATION}', using AUTO")
    set(VIO_GPU_ACCELERATION_MODE "AUTO")
endif()

if(NOT VIO_GPU_ACCELERATION_MODE STREQUAL "OFF")
    find_package(OpenCL QUIET)
else()
    set(OpenCL_FOUND FALSE)
endif()

if(OpenCL_FOUND)
    message(STATUS "OpenCL: found; GPU backend will be compiled")
else()
    message(STATUS "OpenCL: not found; building CPU-only vision backend")
endif()

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
            Threads::Threads
            TBB::tbb
    )

    if(OpenCL_FOUND)
        if(TARGET OpenCL::OpenCL)
            target_link_libraries(${target_name} PRIVATE OpenCL::OpenCL)
        else()
            target_include_directories(${target_name} PRIVATE ${OpenCL_INCLUDE_DIRS})
            target_link_libraries(${target_name} PRIVATE ${OpenCL_LIBRARIES})
        endif()
        target_compile_definitions(${target_name} PRIVATE VIO_HAVE_OPENCL)
    endif()
endfunction()

# =========================
# CANONICAL VIO APP
# =========================
add_executable(vio
        main.cpp

        # approved frontend + backend
        src/frontend/feature_extractor.cpp
        src/frontend/feature_tracker.cpp
        src/frontend/pivot_frame.cpp
        src/frontend/vision_compute_backend.cpp
        src/frontend/visual_frontend.cpp

        # approved pipeline
        src/pipeline/vio_pipeline.cpp

        # approved geometry
        src/geometry/triangulator.cpp
        src/geometry/pnp_solver.cpp
        src/geometry/landmark_map.cpp

        # approved imu
        src/imu/imu_processor.cpp

        # approved io
        src/io/dataset_loader.cpp
        src/io/data_streamer.cpp

        # visualization
        src/visualization/points_video_renderer.cpp
        src/visualization/rerun_visualizer.cpp
)

configure_vio_target(vio)

if(ENABLE_RERUN_VISUALIZATION)
    target_compile_definitions(vio PRIVATE VIO_ENABLE_RERUN_VISUALIZATION)
endif()
