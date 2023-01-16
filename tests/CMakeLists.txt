cmake_minimum_required(VERSION 3.10)

set(PROJECT_TEST_NAME "${PROJECT_NAME}_TEST")

project(${PROJECT_TEST_NAME} LANGUAGES CXX)

# set compiler flags
add_compile_options(-Wall -Wextra -pedantic -O3 -std=c++17)

add_subdirectory(core)
add_subdirectory(util)

add_executable(${PROJECT_TEST_NAME} mainTest.cpp)
target_link_libraries(${PROJECT_TEST_NAME} PUBLIC gtest_main RingPolymerTest NormalModesTest PotentialTest
                                                  UtilTests)

add_test(NAME ${PROJECT_TEST_NAME} COMMAND ${PROJECT_TEST_NAME})
