cmake_minimum_required(VERSION 3.17)
project(handobject_dataset_creator)

set(CMAKE_CXX_STANDARD 20)

# Azure Kinect SDK
find_package(k4a REQUIRED)
find_package(k4arecord REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIRS} )
add_definitions( ${EIGEN3_DEFINITIONS} )

# Boost
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories( ${Boost_INCLUDE_DIR} )

# OpenCV
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# PCL
find_package(PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIR})
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable( scene_viewer ${SOURCE_DIR}/scene_viewer.cpp )
target_link_libraries ( scene_viewer k4a k4arecord Boost::program_options ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_executable( handobject_registration ${SOURCE_DIR}/handobject_registration.cpp)
target_link_libraries( handobject_registration k4a k4arecord Boost::program_options ${OpenCV_LIBS} ${PCL_LIBRARIES} )
