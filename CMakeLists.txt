cmake_minimum_required(VERSION 3.16) # tested with CMake 4.2.1 on Ubuntu 20.04 / ROS2 Foxy

# Prefer CUDA 12.8 toolchain before project() to force correct nvcc
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc CACHE FILEPATH "CUDA 12.8 nvcc")
endif()
if(NOT DEFINED CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT /usr/local/cuda-12.8 CACHE PATH "CUDA 12.8 root")
endif()

project(pct_planner_cpp_port LANGUAGES CXX CUDA)

# Limit CUDA arch to current GPU (RTX 5060 Ti ~ SM 86) to avoid default sm_30 error
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PCL REQUIRED COMPONENTS io)
find_package(pcl_conversions REQUIRED)

# 如果未显式指定 GTSAM_DIR，则尝试使用项目内置的安装路径
if(NOT GTSAM_DIR)
  set(GTSAM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/planner_lib/3rdparty/gtsam-4.1.1/install/lib/cmake/GTSAM)
endif()
find_package(GTSAM REQUIRED)

set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/planner_lib)
set(GTSAM_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/planner_lib/3rdparty/gtsam-4.1.1/install/lib)
set(METIS_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/planner_lib/3rdparty/gtsam-4.1.1/install/lib)
set(RPATH_ORIGIN_PLANNER "\$ORIGIN/../../../planner_lib")
set(RPATH_ORIGIN_GTSAM "\$ORIGIN/../../../planner_lib/3rdparty/gtsam-4.1.1/install/lib")
set(RPATH_ORIGIN_LIB "\$ORIGIN/..")

# 嵌入运行时查找路径，避免每次手动导出 LD_LIBRARY_PATH
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH "${LIB_DIR};${GTSAM_LIB_DIR};${METIS_LIB_DIR};${RPATH_ORIGIN_PLANNER};${RPATH_ORIGIN_GTSAM};${RPATH_ORIGIN_LIB}")
set(CMAKE_INSTALL_RPATH "${LIB_DIR};${GTSAM_LIB_DIR};${METIS_LIB_DIR};${RPATH_ORIGIN_PLANNER};${RPATH_ORIGIN_GTSAM};${RPATH_ORIGIN_LIB}")

link_directories(${GTSAM_LIB_DIR})

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${LIB_DIR}/src
  ${GTSAM_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

link_directories(${LIB_DIR})

add_library(tomography_cuda STATIC src/tomography_cuda.cu)
target_include_directories(tomography_cuda PUBLIC include ${EIGEN3_INCLUDE_DIR})
target_link_libraries(tomography_cuda CUDA::cudart)

add_executable(tomography_node src/tomography_node.cpp)
ament_target_dependencies(tomography_node rclcpp sensor_msgs std_msgs)
target_link_libraries(tomography_node Eigen3::Eigen tomography_cuda)

add_executable(planner_node src/planner_node.cpp)
ament_target_dependencies(planner_node rclcpp std_msgs geometry_msgs nav_msgs)
target_link_libraries(planner_node
  Eigen3::Eigen
  ${LIB_DIR}/libele_planner_lib.so
  ${LIB_DIR}/liba_star_search.so
  ${LIB_DIR}/libgpmp_optimizer.so
  ${LIB_DIR}/libmap_manager.so
  ${LIB_DIR}/libcommon_smoothing.so
  gtsam
)

add_executable(planner_direct_node src/planner_direct_node.cpp)
ament_target_dependencies(planner_direct_node rclcpp std_msgs geometry_msgs nav_msgs)
target_link_libraries(planner_direct_node
  Eigen3::Eigen
  ${LIB_DIR}/libele_planner_lib.so
  ${LIB_DIR}/liba_star_search.so
  ${LIB_DIR}/libgpmp_optimizer.so
  ${LIB_DIR}/libmap_manager.so
  ${LIB_DIR}/libcommon_smoothing.so
  gtsam
)

add_executable(pcd_publisher src/pcd_publisher.cpp)
ament_target_dependencies(pcd_publisher rclcpp sensor_msgs pcl_conversions)
target_link_libraries(pcd_publisher
  Eigen3::Eigen
  ${PCL_LIBRARIES}
)


target_include_directories(planner_node PRIVATE include)
target_include_directories(tomography_node PRIVATE include)
target_include_directories(planner_direct_node PRIVATE include)
target_include_directories(pcd_publisher PRIVATE include)


install(TARGETS
  tomography_node
  planner_node
  planner_direct_node
  pcd_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY rsc/ DESTINATION share/${PROJECT_NAME}/rsc)

# 安装依赖动态库到本包 lib，便于运行时就近加载
set(PLANNER_LIBS
  ${LIB_DIR}/libele_planner_lib.so
  ${LIB_DIR}/liba_star_search.so
  ${LIB_DIR}/libgpmp_optimizer.so
  ${LIB_DIR}/libmap_manager.so
  ${LIB_DIR}/libcommon_smoothing.so
)

set(VENDORED_GTSAM_LIBS
  ${GTSAM_LIB_DIR}/libgtsam.so
  ${GTSAM_LIB_DIR}/libgtsam_unstable.so
  ${GTSAM_LIB_DIR}/libmetis-gtsam.so
)

install(FILES ${PLANNER_LIBS} ${VENDORED_GTSAM_LIBS} DESTINATION lib)

ament_package()
