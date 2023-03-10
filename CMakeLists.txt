cmake_minimum_required(VERSION 3.16)
project(wbee)

set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)
include_directories(./include)

set(link_lib glog pthread gtest)
set(link_math_lib armadillo blas lapack)

# 设置输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

add_executable(wbee main.cpp)

target_include_directories(wbee PUBLIC /usr/include/armadillo_bits)
target_include_directories(wbee PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(wbee ${link_lib} ${link_math_lib})

enable_testing()
add_subdirectory(test)