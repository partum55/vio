cmake_minimum_required(VERSION 3.16)
project(vio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc features2d calib3d highgui dnn)

add_executable(vio_viewer
    src/main.cpp
    src/data_generator.cpp
    src/data_loader.cpp
    src/pose_detection.cpp
    src/viewer.cpp
    src/stereo_triangulation.cpp
)

target_include_directories(vio_viewer PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vio_viewer PRIVATE
    Eigen3::Eigen
    pango_core
    pango_display
    pango_opengl
    OpenGL::GL
    ${OpenCV_LIBS}
)

# Test executable (no Pangolin/viewer dependency)
add_executable(test_triangulation
    tests/test_triangulation.cpp
    src/stereo_triangulation.cpp
)

target_include_directories(test_triangulation PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_triangulation PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

add_executable(test_pose_triangulation
    tests/test_pose_triangulation.cpp
    src/stereo_triangulation.cpp
)

target_include_directories(test_pose_triangulation PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_pose_triangulation PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
)
