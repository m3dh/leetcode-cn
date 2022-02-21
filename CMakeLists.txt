cmake_minimum_required(VERSION 3.5.0)
project(lccn VERSION 0.0.1)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

find_package(PkgConfig REQUIRED)
pkg_search_module(glog REQUIRED libglog)
find_package(gflags REQUIRED)
find_package(folly REQUIRED)

# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fcoroutines")

set(SOURCES
    c2/main.cpp
)

add_executable(lccn ${SOURCES})
target_link_libraries(lccn
    PRIVATE gflags
    ${glog_LIBRARIES}
    PRIVATE Folly::folly
)
target_include_directories(lccn
    PRIVATE ${gflags_INCLUDE_DIR}
    PRIVATE ${glog_INCLUDE_DIR}
    PRIVATE ${folly_INCLUDE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/inc
)
