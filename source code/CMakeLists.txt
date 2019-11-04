cmake_minimum_required(VERSION 2.8)
project(CHENG12_DISSERTATION_VEHICLE_COUNTING2)

add_compile_options(-std=c++14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})

link_directories(${OpenCV_LIBRARY_DIR})

include_directories(${VEHICLE_COUNTING_SOURCE_DIR}/src)
add_subdirectory(src)
