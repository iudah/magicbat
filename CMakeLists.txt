cmake_minimum_required(VERSION 3.14)
project(tensor C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wpedantic -Wextra -O2)

set(target tensor)
add_library(${target} STATIC 
src/info/tensor_ndims.c
src/info/tensor_num_elements.c
src/info/tensor_shape.c
src/lifecycle/tensor_destroy.c
src/lifecycle/tensor_new.c
src/math/tensor_add.c
src/math/tensor_sub.c
src/math/tensor_sum.c
src/math/tensor_mul.c
src/math/tensor_matmul.c
src/math/tensor_div.c
src/math/tensor_exp.c
src/math/tensor_softmax.c
src/math/tensor_binary_op.c
src/math/tensor_max.c
src/ops/tensor_fill.c
src/ops/tensor_reshape.c
src/access/tensor_get.c
src/access/tensor_set.c
src/access/tensor_index_out_of_bound.c
)

target_include_directories(tensor PUBLIC include)
target_include_directories(tensor PRIVATE src)

# Test 
add_executable(test_basic tests/tensor_basic_test.c)
target_link_libraries(test_basic PRIVATE ${target})

add_executable(test_add tests/tensor_add_test.c)
target_link_libraries(test_add PRIVATE ${target})

add_executable(test_sub tests/tensor_sub_test.c)
target_link_libraries(test_sub PRIVATE ${target})

add_executable(test_mul tests/tensor_mul_test.c)
target_link_libraries(test_mul PRIVATE ${target})

add_executable(test_matmul tests/tensor_matmul_test.c)
target_link_libraries(test_matmul PRIVATE ${target})

add_executable(test_div tests/tensor_div_test.c)
target_link_libraries(test_div PRIVATE ${target})

add_executable(test_broadcast tests/tensor_broadcast_test.c)
target_link_libraries(test_broadcast PRIVATE ${target})

add_executable(test_softmax tests/tensor_softmax_test.c)
target_link_libraries(test_softmax PRIVATE ${target} m)

add_executable(test_max tests/tensor_max_test.c)
target_link_libraries(test_max PRIVATE ${target} m)
