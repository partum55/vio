cmake_minimum_required(VERSION 3.16)
project(vio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(vio_viewer
    src/main.cpp
    src/data_generator.cpp
    src/data_loader.cpp
    src/viewer.cpp
)

target_include_directories(vio_viewer PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(vio_viewer PRIVATE
    Eigen3::Eigen
    pango_core
    pango_display
    pango_opengl
    OpenGL::GL
)
