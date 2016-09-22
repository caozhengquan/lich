cmake_minimum_required(VERSION 2.8.0)
project(lich)
include_directories(./)
include_directories(./thirdparty/gtest/include)
include_directories(./thirdparty/glog/include)
include_directories(./thirdparty/gflags/include)
include_directories(./thirdparty/protobuf3/include)
include_directories(./thirdparty/benchmark/include)
include_directories(./thirdparty/openblas/include)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g -pthread")

FILE(GLOB PROTO_INPUT ${CMAKE_SOURCE_DIR}/proto/filler_param.proto)


# common source files
set (SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/lich/lib/math.cc"
    "${CMAKE_SOURCE_DIR}/lich/src/tensor.cc"
    "${CMAKE_SOURCE_DIR}/lich/src/layers/dense_layer.cc"
    "${CMAKE_SOURCE_DIR}/lich/src/layers/data_layer.cc"
    "${CMAKE_SOURCE_DIR}/lich/src/layers/embedding_layer.cc"
    "${CMAKE_SOURCE_DIR}/lich/src/layers/softmax_layer.cc"
    "${CMAKE_SOURCE_DIR}/lich/src/layers/softmax_cross_entropy_loss_layer.cc"
    "${CMAKE_SOURCE_DIR}/lich/proto/filler_param.pb.cc"
    "${CMAKE_SOURCE_DIR}/lich/proto/layer_param.pb.cc"
    )

# test source files 
set (TEST_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/lich/test/test_main.cc"
    "${CMAKE_SOURCE_DIR}/lich/test/tensor_test.cc"
    "${CMAKE_SOURCE_DIR}/lich/test/embedding_layer_test.cc"
    "${CMAKE_SOURCE_DIR}/lich/test/filler_test.cc"
    "${CMAKE_SOURCE_DIR}/lich/test/dense_layer_test.cc"
    "${CMAKE_SOURCE_DIR}/lich/test/data_layer_test.cc"
    "${CMAKE_SOURCE_DIR}/lich/test/softmax_layer_test.cc"
    "${CMAKE_SOURCE_DIR}/lich/test/softmax_cross_entropy_loss_layer_test.cc"
     )

# test case
add_executable (lichtest
  ${SOURCE_FILES}
  ${TEST_SOURCE_FILES})

# link thirdparty libs
set (THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/thirdparty)

target_link_libraries (lichtest
    "${THIRDPARTY_DIR}/libgtest.a"
    "${THIRDPARTY_DIR}/libglog.a"
    "${THIRDPARTY_DIR}/libgflags.a"
    "${THIRDPARTY_DIR}/libprotobuf.a"
    "${THIRDPARTY_DIR}/libblas.a"
    )

