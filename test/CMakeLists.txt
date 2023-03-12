find_package(GTest REQUIRED)
#设置可调式、
# $ cmake -DCMAKE_BUILD_TYPE=Debug ..
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

aux_source_directory(../test DIR_TEST)
set(link_lib glog gtest pthread  )
set(link_math_lib armadillo blas lapack)

# 添加源文件
aux_source_directory(../source/data DIR_DATA)
aux_source_directory(../source/layer/abstract DIR_LAYER_ABSTRACT)
aux_source_directory(../source/layer/details DIR_LAYER_DETAILS)
aux_source_directory(../source/ops DIR_OPS)
aux_source_directory(../source/runtime DIR_RUNTIME)

add_executable(test_wbee ${DIR_TEST} ${DIR_DATA} ${DIR_LAYER_ABSTRACT} ${DIR_LAYER_DETAILS} ${DIR_OPS} ${DIR_RUNTIME})

link_directories(/usr/local/lib/)
target_link_libraries(test_wbee ${link_lib} ${link_math_lib})
