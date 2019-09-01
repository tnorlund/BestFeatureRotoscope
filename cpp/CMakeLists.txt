cmake_minimum_required(VERSION 2.8)

project( DisplayImage )
# PATH to OpenCV ( uncomment and provide path if required )
# set(OpenCV_DIR /Users/visionifai/codes/installations/OpenCV4/lib/cmake/opencv4/)


# set(OpenCV_DIR ./opencv-3.3.0/build)
find_package( OpenCV REQUIRED )
find_package( MPI )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories( SYSTEM ${MPI_INCLUDE_PATH} )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( main main.cpp )
target_link_libraries( main ${OpenCV_LIBS} ${MPI_C_LIBRARIES})
