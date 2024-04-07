cmake_minimum_required(VERSION 3.0.2)
project(pointpillar)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++17 -O0 -g")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_87,code=compute_87
)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  vision_msgs
  pcl_ros
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pointpillar
 CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


include_directories(
    $ENV{CUDA_Inc}
    $ENV{TensorRT_Inc}
    src
	  include/pointpillar

)

link_directories(
    $ENV{CUDA_Lib}
    $ENV{TensorRT_Lib}
    build
)

file(GLOB_RECURSE CORE_FILES 
  src/*.cu 
  src/*.cpp
  include/pointpillar/tensor.cu
  include/pointpillar/tensorrt.cpp
)
file(GLOB_RECURSE ROS_SRCS "src/*_ros.cpp")
list(REMOVE_ITEM CORE_FILES ${ROS_SRCS})



####################### libpointpillar_core.so ##########################
cuda_add_library(pointpillar_core SHARED 
  	${CORE_FILES}
)
target_link_libraries(pointpillar_core
	libnvinfer.so
)
#######################################################################

####################### pointpillar program #############################
# cuda_add_executable(pointpillar 
# 	main.cpp
# )

# target_link_libraries(pointpillar
# 	pointpillar_core ${catkin_LIBRARIES}
# )

#######################################################################

####################### ros pointpillar program #######################


cuda_add_executable(pointpillar_node
pointpillar_node.cpp ${ROS_SRCS} 

)
add_dependencies(pointpillar_node ${catkin_EXPORTED_TARGETS})

target_link_libraries(pointpillar_node
	pointpillar_core ${catkin_LIBRARIES} libyaml-cpp.so  ${OpenCV_LIBRARIES} 
)
#######################################################################