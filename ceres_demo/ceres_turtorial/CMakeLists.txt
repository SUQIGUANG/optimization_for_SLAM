cmake_minimum_required(VERSION 3.13)
project(ceres_turtorial)

set(CMAKE_CXX_STANDARD 11)

# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

add_executable(ceres_turtorial main.cpp)

# 与Ceres和链接
target_link_libraries( ceres_turtorial ${CERES_LIBRARIES} )
