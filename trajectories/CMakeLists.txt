cmake_minimum_required(VERSION 2.8)
project( trajectories )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( trajectories trajectories.cpp )
target_link_libraries( trajectories ${OpenCV_LIBS} )