cmake_minimum_required(VERSION 3.16)
project(project)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

# executable для main.cpp
add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE ${OpenCV_LIBS})

# executable для algo.cpp
add_executable(algo algo.cpp)
target_link_libraries(algo PRIVATE ${OpenCV_LIBS})

# executable для algo_with_cam0.cpp
add_executable(algo_with_cam0 algo_with_cam0.cpp)
target_link_libraries(algo_with_cam0 PRIVATE ${OpenCV_LIBS})
