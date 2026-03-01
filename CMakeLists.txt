cmake_minimum_required(VERSION 3.16)
project(project)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)


add_executable(project main.cpp)
target_link_libraries(project PRIVATE ${OpenCV_LIBS})
