cmake_minimum_required(VERSION 3.14)
project(tensor C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wpedantic -Wextra -O2)

set(target tensor)

add_library(${target} STATIC lifecycle/tensor_new.c lifecycle/tensor_destroy.c)

target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Test 
add_executable(test_basic tests/tensor_basic_test.c)
target_link_libraries(test_basic PRIVATE ${target})