cmake_minimum_required(VERSION 3.14)
project(tensor C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wpedantic -Wextra -O0 -g)

set(SOURCES 
pcg/extras/entropy.c
pcg/src/pcg-advance-128.c
pcg/src/pcg-advance-16.c
pcg/src/pcg-advance-32.c
pcg/src/pcg-advance-64.c
pcg/src/pcg-advance-8.c
pcg/src/pcg-global-32.c
pcg/src/pcg-global-64.c
pcg/src/pcg-output-128.c
pcg/src/pcg-output-16.c
pcg/src/pcg-output-32.c
pcg/src/pcg-output-64.c
pcg/src/pcg-output-8.c
pcg/src/pcg-rngs-128.c
pcg/src/pcg-rngs-16.c
pcg/src/pcg-rngs-32.c
pcg/src/pcg-rngs-64.c
pcg/src/pcg-rngs-8.c
src/info/tensor_ndims.c
src/info/tensor_num_elements.c
src/info/tensor_shape.c
src/lifecycle/tensor_destroy.c
src/lifecycle/tensor_new.c
src/math/tensor_add.c
src/math/tensor_matmul.c
src/math/tensor_mul.c
src/math/tensor_div.c
src/math/tensor_sub.c
src/math/tensor_sum.c
src/math/tensor_binary_op.c
src/math/tensor_exp.c
src/math/tensor_max.c
src/math/tensor_relu.c
src/math/tensor_softmax.c
src/math/tensor_cross_entropy.c
src/math/tensor_gather.c
src/math/tensor_log.c
src/math/tensor_log_sum_exp.c
src/math/tensor_negate.c
src/math/tensor_transpose.c
src/math/tensor_tanh.c
src/ops/tensor_fill.c
src/ops/tensor_reshape.c
src/ops/tensor_copy_data.c
src/ops/tensor_scalar.c
src/ops/tensor_zero.c
src/ops/tensor_slice.c
src/ops/tensor_concat.c
src/access/tensor_get.c
src/access/tensor_set.c
src/access/tensor_index_out_of_bound.c
src/layers/rnn/rnn_cell_track.c
src/layers/rnn/rnn_cell_new.c
src/layers/rnn/rnn_cell_forward.c
src/layers/rnn/rnn_cell_kernel.c
src/layers/rnn/rnn_cell_destroy.c
src/layers/linear/linear_layer_destroy.c
src/layers/linear/linear_layer_forward.c
src/layers/linear/linear_layer_kernel.c
src/layers/linear/linear_layer_new.c
src/layers/linear/linear_layer_track.c
src/optimizers/sgd/sgd_destroy.c
src/optimizers/sgd/sgd_new.c
src/optimizers/sgd/sgd_optimize.c
src/var/var_add.c
src/var/var_backward.c
src/var/var_destroy.c
src/var/var_div.c
src/var/var_matmul.c
src/var/var_mul.c
src/var/var_new.c
src/var/var_relu.c
src/var/var_sub.c
src/var/var_sum.c
src/var/var_zero_grad.c
src/var/var_tanh.c
src/var/var_copy_data.c
src/var/var_concat.c
)

set (HEADERS
include/tensor.h
include/adt/layers/layers_prot.h
include/adt/optimizers/sgd_prot.h
include/adt/tensor/tensor_prot.h
include/adt/type_alias.h
include/adt/var/var_prot.h
include/layers/linear_layer.h
include/layers/rnn/rnn_cell.h
include/optimizers/sgd.h
include/var/var.h
pcg/extras/entropy.h
pcg/extras/pcg_spinlock.h
pcg/include/pcg_variants.h
src/lifecycle/tensor_memory.h
src/lifecycle/data_storage.h
src/math/tensor_shapes_equal.h
src/math/tensor_binary_op.h
src/math/tensor_odometer.h
src/math/tensor_shapes_broadcast.h
src/math/tensor_shapes_broadcast_old.h
src/access/tensor_increment_index.h
src/access/tensor_ndim_flat_index.h
)

set(INCLUDE_DIRS
include/
include/adt
include/adt/layers
include/adt/optimizers
include/adt/tensor
include/adt/var
include/layers
include/layers/rnn
include/layers/linear
include/optimizers
include/var
pcg
pcg/extras
pcg/include
pcg/src
src/
src/info
src/lifecycle
src/math
src/ops
src/access
src/layers
src/layers/rnn
src/layers/linear
src/optimizers
src/optimizers/sgd
src/var
)

set(target tensor)
add_library(${target} STATIC ${SOURCES})
target_sources(${target} PRIVATE ${HEADERS})

target_include_directories(tensor PUBLIC include)
target_include_directories(tensor PRIVATE ${INCLUDE_DIRS})

# Test 
add_executable(test_basic tests/tensor_basic_test.c)
target_link_libraries(test_basic PRIVATE ${target})
target_include_directories(test_basic PRIVATE ${INCLUDE_DIRS})
target_sources(test_basic PRIVATE ${HEADERS})

add_executable(test_add tests/tensor_add_test.c)
target_link_libraries(test_add PRIVATE ${target})
target_include_directories(test_add PRIVATE ${INCLUDE_DIRS})
target_sources(test_add PRIVATE ${HEADERS})

add_executable(test_sub tests/tensor_sub_test.c)
target_link_libraries(test_sub PRIVATE ${target})
target_include_directories(test_sub PRIVATE ${INCLUDE_DIRS})
target_sources(test_sub PRIVATE ${HEADERS})

add_executable(test_mul tests/tensor_mul_test.c)
target_link_libraries(test_mul PRIVATE ${target})
target_include_directories(test_mul PRIVATE ${INCLUDE_DIRS})
target_sources(test_mul PRIVATE ${HEADERS})

add_executable(test_matmul tests/tensor_matmul_test.c)
target_link_libraries(test_matmul PRIVATE ${target})
target_include_directories(test_matmul PRIVATE ${INCLUDE_DIRS})
target_sources(test_matmul PRIVATE ${HEADERS})

add_executable(test_div tests/tensor_div_test.c)
target_link_libraries(test_div PRIVATE ${target})
target_include_directories(test_div PRIVATE ${INCLUDE_DIRS})
target_sources(test_div PRIVATE ${HEADERS})

add_executable(test_broadcast tests/tensor_broadcast_test.c)
target_link_libraries(test_broadcast PRIVATE ${target})
target_include_directories(test_broadcast PRIVATE ${INCLUDE_DIRS})
target_sources(test_broadcast PRIVATE ${HEADERS})

add_executable(test_softmax tests/tensor_softmax_test.c)
target_link_libraries(test_softmax PRIVATE ${target} m)
target_include_directories(test_softmax PRIVATE ${INCLUDE_DIRS})
target_sources(test_softmax PRIVATE ${HEADERS})

add_executable(test_max tests/tensor_max_test.c)
target_link_libraries(test_max PRIVATE ${target} m)
target_include_directories(test_max PRIVATE ${INCLUDE_DIRS})
target_sources(test_max PRIVATE ${HEADERS})

add_executable(test_train_mlp tests/train_mlp.c)
target_link_libraries(test_train_mlp PRIVATE tensor m)
target_include_directories(test_train_mlp PRIVATE ${INCLUDE_DIRS})
target_sources(test_train_mlp PRIVATE ${HEADERS})

add_executable(test_train_mlp_auto tests/train_mlp_auto.c)
target_link_libraries(test_train_mlp_auto PRIVATE tensor m)
target_include_directories(test_train_mlp_auto PRIVATE ${INCLUDE_DIRS})
target_sources(test_train_mlp_auto PRIVATE ${HEADERS})

add_executable(test_train_rnn_auto tests/train_rnn_auto.c)
target_link_libraries(test_train_rnn_auto PRIVATE tensor m)
target_include_directories(test_train_rnn_auto PRIVATE ${INCLUDE_DIRS})
target_sources(test_train_rnn_auto PRIVATE ${HEADERS})
