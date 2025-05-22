cmake_minimum_required(VERSION 3.5)
project(camera_cones_detector)


# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3) 
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(common_msgs REQUIRED)

include_directories(
  include/camera_cones_detector
)

add_library(camera_cones_detector src/camera_cones_detector.cpp)

ament_target_dependencies(camera_cones_detector
    rclcpp
    std_msgs
    tf2
    tf2_geometry_msgs
    visualization_msgs
    Eigen3
    # common_msgs
    )

add_executable(camera_cones_detector_node src/camera_cones_detector_node.cpp)

target_link_libraries(camera_cones_detector_node camera_cones_detector)

install(TARGETS
  camera_cones_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config directory
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
