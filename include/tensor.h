#ifndef TENSOR_H
#define TENSOR_H

#include "adt/tensor/tensor_prot.h"
#include "adt/type_alias.h"
#include <stdbool.h>

typedef struct tensor_struct *Tensor;
typedef enum {
  TENSOR_CE_INDEXED = 1 << 1,
  TENSOR_CE_ONE_HOT = 1 << 2,
  TENSOR_CE_PROBS = 1 << 3
} TensorType;

typedef struct {
  u32 start;
  u32 stop;
  u32 step;
} TensorSlice;

typedef struct {
  u32 dest;
  u32 src;
} TensorDimSwap;

f32 tensor_get(Tensor tensor_a, const u32 *index);
bool tensor_index_out_of_bound(Tensor tensor_a, const u32 *index);
u32 tensor_ndims(Tensor tensor);
u32 tensor_num_elements(Tensor tensor);
bool tensor_set(Tensor tensor_a, const u32 *index, f32 value);
const u32 *tensor_shape(Tensor tensor);

Tensor tensor_new(u32 ndims, const u32 *shape);
bool tensor_destroy(Tensor tensor);

Tensor tensor_add(Tensor tensor_a, Tensor tensor_b);
bool tensor_add_inplace(Tensor tensor_a, Tensor tensor_b);
bool tensor_add_scalar_inplace(Tensor tensor_a, f32 f_value);
Tensor tensor_scaled_add(Tensor tensor_a, f32 alpha, Tensor tensor_b);
Tensor tensor_sub(Tensor tensor_a, Tensor tensor_b);
bool tensor_sub_inplace(Tensor tensor_a, Tensor tensor_b);
Tensor tensor_mul(Tensor tensor_a, Tensor tensor_b);
Tensor tensor_div(Tensor tensor_a, Tensor tensor_b);
Tensor tensor_divisor_backward(Tensor divisor, Tensor grad);
Tensor tensor_matmul(Tensor tensor_a, Tensor tensor_b);
Tensor tensor_matmul_wrt_a(Tensor tensor_grad, Tensor tensor_b);
Tensor tensor_matmul_wrt_b(Tensor tensor_a, Tensor tensor_grad);
Tensor tensor_bmm(Tensor tensor_a, Tensor tensor_b);
Tensor tensor_bmm_wrt_a(Tensor tensor_grad, Tensor tensor_b);
Tensor tensor_bmm_wrt_b(Tensor tensor_a, Tensor tensor_grad);
Tensor tensor_sum_axis(Tensor tensor, i32 axis);
f32 tensor_sum_all(Tensor tensor);
Tensor tensor_sum_to_shape(Tensor tensor, u32 ndims, u32 *shape);
Tensor tensor_max_axis(Tensor tensor, i32 axis);
f32 tensor_max_all(Tensor tensor);
Tensor tensor_gather_axis(Tensor tensor, Tensor indices, i32 axis);

Tensor tensor_transpose(Tensor tensor);
Tensor tensor_transpose_dims(Tensor tensor, u32 nswap, TensorDimSwap swaps[]);
Tensor tensor_relu(Tensor tensor);
Tensor tensor_relu_backward(Tensor tensor, Tensor grad);
Tensor tensor_tanh(Tensor tensor);
Tensor tensor_tanh_backward(Tensor tensor, Tensor grad);
Tensor tensor_negate(Tensor tensor);
Tensor tensor_exp(Tensor tensor);
Tensor tensor_log(Tensor tensor);
Tensor tensor_softmax_all(Tensor tensor);
Tensor tensor_softmax_axis(Tensor tensor, i32 axis);
Tensor tensor_log_sum_exp_all(Tensor tensor);
Tensor tensor_log_sum_exp_axis(Tensor tensor, i32 axis);

Tensor tensor_scalar(f32 value);
Tensor tensor_view(Tensor src);
bool tensor_fill(Tensor tensor, f32 value);
bool tensor_masked_fill(Tensor tensor, Tensor mask, f32 value);
Tensor tensor_zero(u32 ndims, const u32 *shape);
Tensor tensor_reshape(Tensor var_a, u32 ndims, const u32 *shape);
bool tensor_copy_data(Tensor dst, Tensor src);
Tensor tensor_concat(u32 n_tensor, Tensor *tensor, u32 axis);
Tensor tensor_slice(Tensor tensor, TensorSlice slices[MAX_DIMS]);
void tensor_random(Tensor tensor);
void tensor_random_bound(Tensor tensor, f32 lower_bound, f32 upper_bound);
void tensor_xavier(Tensor tensor, f32 fanin, f32 fanout);

#endif
