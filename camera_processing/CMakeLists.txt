cmake_minimum_required(VERSION 2.8)
project( camera_processing )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( camera_processing main.cpp )
target_link_libraries( camera_processing ${OpenCV_LIBS} )