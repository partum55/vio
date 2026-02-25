cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME shi_tomasi)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/shi_tomasi.cpp
        src/shi_tomasi_main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE include/)

target_link_libraries(shi_tomasi ${OpenCV_LIBS})
