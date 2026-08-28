cmake_minimum_required(VERSION 3.14)
project(tensor C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wpedantic -Wextra -O0 -g -DNDEBUG)

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
src/lifecycle/tensor_arena.c
src/lifecycle/tensor_memory.c
src/lifecycle/tensor_serialize.c
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
src/math/tensor_bmm.c
src/math/tensor_mean.c
src/math/tensor_variance.c
src/math/tensor_mse.c
src/math/tensor_bmm_bt.c
src/math/tensor_bmm_at.c
src/math/tensor_trinary_op.c
src/ops/tensor_fill.c
src/ops/tensor_reshape.c
src/ops/tensor_copy_data.c
src/ops/tensor_scalar.c
src/ops/tensor_zero.c
src/ops/tensor_slice.c
src/ops/tensor_concat.c
src/ops/tensor_random.c
src/ops/tensor_to_contiguous.c
src/ops/tensor_positional_encoding.c
src/ops/tensor_embedding.c
src/access/tensor_get.c
src/access/tensor_set.c
src/access/tensor_index_out_of_bound.c
src/layers/rnn/rnn_cell_destroy.c
src/layers/rnn/rnn_cell_forward.c
src/layers/rnn/rnn_cell_kernel.c
src/layers/rnn/rnn_cell_new.c
src/layers/rnn/rnn_cell_track.c
src/layers/rnn/rnn_cell_serialize.c
src/layers/transformer/transformer_enc_block_kernel.c
src/layers/transformer/transformer_enc_block_forward.c
src/layers/transformer/transformer_enc_block_new.c
src/layers/transformer/transformer_enc_block_destroy.c
src/layers/transformer/transformer_enc_block_serialize.c
src/layers/layer_norm/layer_norm.c
src/layers/residual_conn/residual.c
src/layers/bin/layer_serialize.c
src/layers/linear/linear_layer_destroy.c
src/layers/linear/linear_layer_forward.c
src/layers/linear/linear_layer_kernel.c
src/layers/linear/linear_layer_new.c
src/layers/linear/linear_layer_track.c
src/layers/linear/linear_layer_serialize.c
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
src/var/var_concat.c
src/var/var_copy_data.c
src/var/var_layer_norm.c
src/var/var_mha.c
src/var/var_mse.c
src/var/var_bmm.c
src/var/var_softmax.c
src/var/var_mha_causal.c
src/var/var_ce.c
src/var/var_embedding.c
src/var/var_inference.c
mgrind/mgrind.c
)

set (HEADERS
pcg/extras/entropy.h
pcg/extras/pcg_spinlock.h
pcg/include/pcg_variants.h
src/lifecycle/tensor_memory.h
src/lifecycle/data_storage.h
src/lifecycle/tensor_arena.h
src/math/tensor_shapes_equal.h
src/math/tensor_binary_op.h
src/math/tensor_odometer.h
src/math/tensor_shapes_broadcast.h
src/math/tensor_reduce.h
src/math/matmul.h
src/math/matmul_tile.h
src/math/matmul_batched_tile.h
src/math/fused_attention.h
src/math/fused_attention_backward.h
src/math/fused_attention_causal.h
src/math/fused_attention_causal_backward.h
src/math/tensor_trinary_op.h
src/access/tensor_increment_index.h
src/access/tensor_ndim_flat_index.h
include/tensor.h
include/adt/layers/layers_prot.h
include/adt/optimizers/sgd_prot.h
include/adt/tensor/tensor_prot.h
include/adt/type_alias.h
include/adt/var/var_prot.h
include/layers/residual_conn/residual.h
include/layers/rnn/rnn_cell.h
include/layers/linear/linear_layer.h
include/layers/layer_norm/layer_norm.h
include/layers/transformer/transformer_enc_layer.h
include/layers/bin/layer_serialize.h
include/optimizers/sgd.h
include/var/var.h
)

set(INCLUDE_DIRS
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
src/layers/transformer
src/layers/layer_norm
src/layers/residual_conn
src/layers/bin
src/layers/linear
src/optimizers
src/optimizers/sgd
src/var
include/
include/adt
include/adt/layers
include/adt/optimizers
include/adt/tensor
include/adt/var
include/layers
include/layers/residual_conn
include/layers/rnn
include/layers/linear
include/layers/layer_norm
include/layers/transformer
include/layers/bin
include/optimizers
include/var
)

include(ExternalProject)

ExternalProject_Add(
    Libbacktrace
    SOURCE_DIR "${CMAKE_BINARY_DIR}/libbacktrace-src"
    BINARY_DIR  "${CMAKE_BINARY_DIR}/libbacktrace-build"
    DOWNLOAD_COMMAND
        ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/libbacktrace"
            "${CMAKE_BINARY_DIR}/libbacktrace-src"
    PATCH_COMMAND chmod 755 <SOURCE_DIR>/configure
    CONFIGURE_COMMAND
        <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
    BUILD_COMMAND
        make
    INSTALL_COMMAND
        make install
)

ExternalProject_Get_Property(Libbacktrace INSTALL_DIR)
list(APPEND INCLUDE_DIRS "${INSTALL_DIR}/include")
set(BT_INSTALL_DIR ${INSTALL_DIR})

add_library(backtrace STATIC IMPORTED)
set_target_properties(backtrace PROPERTIES
  IMPORTED_LOCATION "${BT_INSTALL_DIR}/lib/libbacktrace.a"
)

if(NOT MSVC)
    add_compile_options(-fsanitize=address -fsanitize=undefined -O0 -g -funwind-tables -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -g -rdynamic -funwind-tables)
else()
    add_compile_options(/fsanitize=address /Zi)
    add_link_options(/INCREMENTAL:NO)
endif()

set(target tensor)
add_library(${target} STATIC ${SOURCES})
target_sources(${target} PRIVATE ${HEADERS})
target_link_libraries(${target} PUBLIC backtrace)

target_include_directories(tensor PUBLIC include)
target_include_directories(tensor PRIVATE ${INCLUDE_DIRS})

# Test 
add_executable(test_basic tests/tensor_basic_test.c)
target_link_libraries(test_basic PRIVATE ${target})
target_include_directories(test_basic PRIVATE ${INCLUDE_DIRS})
target_sources(test_basic PRIVATE ${HEADERS})

add_executable(test_add tests/tensor_add_test.c)
target_link_libraries(test_add PRIVATE ${target} backtrace)
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

add_executable(test_bmm tests/tensor_bmm.c)
target_link_libraries(test_bmm PRIVATE tensor m)
target_include_directories(test_bmm PRIVATE ${INCLUDE_DIRS})
target_sources(test_bmm PRIVATE ${HEADERS})

add_executable(test_mha tests/test_mha_fused.c tests/test_mha_discrete.c tests/test_mha.c)
target_link_libraries(test_mha PRIVATE tensor m)
target_include_directories(test_mha PRIVATE ${INCLUDE_DIRS})
target_sources(test_mha PRIVATE ${HEADERS})

add_executable(test_matmul_other tests/tensor_matmul_other.c)
target_link_libraries(test_matmul_other PRIVATE ${target} m)
target_include_directories(test_matmul_other PRIVATE ${INCLUDE_DIRS})
target_sources(test_matmul_other PRIVATE ${HEADERS})

add_executable(test_layernorm tests/test_layernorm.c)
target_link_libraries(test_layernorm PRIVATE ${target} m)
target_include_directories(test_layernorm PRIVATE ${INCLUDE_DIRS})
target_sources(test_layernorm PRIVATE ${HEADERS})

add_executable(test_crossentropy tests/test_crossentropy.c)
target_link_libraries(test_crossentropy PRIVATE ${target} m)
target_include_directories(test_crossentropy PRIVATE ${INCLUDE_DIRS})
target_sources(test_crossentropy PRIVATE ${HEADERS})

add_executable(train_first_primer tests/train_first_primer.c)
target_link_libraries(train_first_primer PRIVATE ${target} m)
target_include_directories(train_first_primer PRIVATE ${INCLUDE_DIRS})
target_sources(train_first_primer PRIVATE ${HEADERS})

add_executable(train_first_primer_and_infer tests/train_first_primer_and_infer.c)
target_link_libraries(train_first_primer_and_infer PRIVATE ${target} m)
target_include_directories(train_first_primer_and_infer PRIVATE ${INCLUDE_DIRS})
target_sources(train_first_primer_and_infer PRIVATE ${HEADERS})

add_executable(test_save_load tests/save_load_first_primer.c)
target_link_libraries(test_save_load PRIVATE ${target})
target_include_directories(test_save_load PRIVATE ${INCLUDE_DIRS})
target_sources(test_save_load PRIVATE ${HEADERS})

