cmake_minimum_required(VERSION 3.16)
project(project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# ---------- Libraries ----------

add_library(lk_tracker src/lk_tracker.cpp)
target_include_directories(lk_tracker PUBLIC include)
target_link_libraries(lk_tracker PUBLIC ${OpenCV_LIBS})

add_library(lk_tracker_without_pyr src/lk_tracker_without_pyr.cpp)
target_include_directories(lk_tracker_without_pyr PUBLIC include)
target_link_libraries(lk_tracker_without_pyr PUBLIC ${OpenCV_LIBS})

add_library(tracking_vis src/tracking_vis.cpp)
target_include_directories(tracking_vis PUBLIC include)
target_link_libraries(tracking_vis PUBLIC ${OpenCV_LIBS})

add_library(feature_refresh src/feature_refresh.cpp)
target_include_directories(feature_refresh PUBLIC include)
target_link_libraries(feature_refresh PUBLIC ${OpenCV_LIBS})

# ---------- Executables ----------

# базовий
add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE lk_tracker tracking_vis feature_refresh)

# algo з пірамідою
add_executable(algo src/algo.cpp)
target_link_libraries(algo PRIVATE lk_tracker tracking_vis feature_refresh)

add_executable(algo_with_cam0 src/algo_with_cam0.cpp)
target_link_libraries(algo_with_cam0 PRIVATE lk_tracker tracking_vis feature_refresh)

# algo БЕЗ піраміди
add_executable(algo_with_cam0_without_pyr src/algo_with_cam0_without_pyr.cpp)
target_link_libraries(algo_with_cam0_without_pyr PRIVATE lk_tracker_without_pyr tracking_vis feature_refresh)