cmake_minimum_required(VERSION 3.16)
project(project)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

# executable для main.cpp
add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE ${OpenCV_LIBS})

# executable для algo.cpp
add_executable(algo_app algo.cpp)
target_link_libraries(algo_app PRIVATE ${OpenCV_LIBS})

# executable для algo_withoutpyr.cpp
add_executable(algo_without algo_withoutpyr.cpp)
target_link_libraries(algo_without PRIVATE ${OpenCV_LIBS})