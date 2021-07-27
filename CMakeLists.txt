cmake_minimum_required(VERSION 3.0.2)
project(CMU_Camera_Projection)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  vision_msgs
  sensor_msgs
  CMU_Mask_R_CNN
)
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  trunk.msg
  predictions.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  vision_msgs
  sensor_msgs
  CMU_Mask_R_CNN
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES CMU_Mask_R_CNN
  CATKIN_DEPENDS message_runtime vision_msgs sensor_msgs CMU_Mask_R_CNN
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)