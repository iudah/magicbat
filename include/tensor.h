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

f32 tensor_get(const Tensor var_a, const u32 *index);
bool tensor_index_out_of_bound(const Tensor var_a, const u32 *index);
u32 tensor_ndims(const Tensor t);
u32 tensor_num_elements(const Tensor t);
bool tensor_set(const Tensor var_a, const u32 *index, f32 value);
const u32 *tensor_shape(const Tensor t);

Tensor tensor_new(const u32 ndims, const u32 *shape);
bool tensor_destroy(Tensor tensor);

Tensor tensor_add(Tensor tensor_a, Tensor tensor_b);
bool tensor_add_inplace(Tensor tensor_a, Tensor tensor_b);
bool tensor_add_scalar_inplace(Tensor tensor_a, f32 f_value);
Tensor tensor_scaled_add(Tensor tensor_a, f32 alpha, Tensor tensor_b);
Tensor tensor_sub(const Tensor var_a, const Tensor var_b);
bool tensor_sub_inplace(Tensor var_a, const Tensor var_b);
Tensor tensor_mul(const Tensor var_a, const Tensor var_b);
Tensor tensor_div(const Tensor var_a, const Tensor var_b);
Tensor tensor_divisor_backward(const Tensor divisor, const Tensor grad);
Tensor tensor_matmul(const Tensor var_a, const Tensor var_b);
Tensor tensor_sum_axis(const Tensor var_a, i32 axis);
f32 tensor_sum_all(const Tensor t);
Tensor tensor_sum_to_shape(Tensor var_a, u32 ndims, u32 *shape);
Tensor tensor_max_axis(const Tensor var_a, i32 axis);
f32 tensor_max_all(const Tensor t);
Tensor tensor_gather_axis(const Tensor var_a, const Tensor indices, i32 axis);

Tensor tensor_transpose(const Tensor t);
Tensor tensor_relu(const Tensor t);
Tensor tensor_relu_backward(const Tensor var_a, const Tensor grad);
Tensor tensor_tanh(const Tensor t);
Tensor tensor_tanh_backward(const Tensor var_a, const Tensor grad);
Tensor tensor_negate(const Tensor t);
Tensor tensor_exp(const Tensor t);
Tensor tensor_log(const Tensor t);
Tensor tensor_softmax_all(const Tensor t);
Tensor tensor_softmax_axis(const Tensor var_a, i32 axis);
Tensor tensor_log_sum_exp_all(const Tensor t);
Tensor tensor_log_sum_exp_axis(const Tensor var_a, i32 axis);

Tensor tensor_scalar(f32 value);
bool tensor_fill(const Tensor var_a, f32 value);
Tensor tensor_zero(u32 ndims, const u32 *shape);
Tensor tensor_reshape(const Tensor var_a, u32 ndims, const u32 *shape);
bool tensor_copy_data(Tensor dst, const Tensor src);
Tensor tensor_concat(u32 n_tensor, Tensor *tensor, u32 axis);
Tensor tensor_slice(Tensor tensor, TensorSlice slices[MAX_DIMS]);

#endif
