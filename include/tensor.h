#ifndef TENSOR_H
#define TENSOR_H

#include "adt/type_alias.h"
#include <stdbool.h>

typedef struct tensor_struct *Tensor;
typedef enum {
  TENSOR_CE_INDEXED = 1 << 1,
  TENSOR_CE_ONE_HOT = 1 << 2,
  TENSOR_CE_PROBS = 1 << 3
} TensorType;

f32 tensor_get(const Tensor t, const u32 *index);
bool tensor_index_out_of_bound(const Tensor t, const u32 *index);
u32 tensor_ndims(const Tensor t);
u32 tensor_num_elements(const Tensor t);
bool tensor_set(const Tensor t, const u32 *index, f32 value);
const u32 *tensor_shape(const Tensor t);

Tensor tensor_new(const u32 ndims, const u32 *shape);
bool tensor_destroy(Tensor t);

Tensor tensor_add(const Tensor t, const Tensor s);
Tensor tensor_scaled_add(const Tensor t, f32 alpha, const Tensor s);
Tensor tensor_sub(const Tensor t, const Tensor s);
Tensor tensor_mul(const Tensor t, const Tensor s);
Tensor tensor_div(const Tensor t, const Tensor s);
Tensor tensor_matmul(const Tensor t, const Tensor s);
Tensor tensor_sum_axis(const Tensor t, i32 axis);
f32 tensor_sum_all(const Tensor t);
Tensor tensor_max_axis(const Tensor t, i32 axis);
f32 tensor_max_all(const Tensor t);
Tensor tensor_gather_axis(const Tensor t, const Tensor indices, i32 axis);

Tensor tensor_transpose(const Tensor t);
Tensor tensor_relu(const Tensor t);
Tensor tensor_relu_backward(const Tensor t, const Tensor grad);
Tensor tensor_negate(const Tensor t);
Tensor tensor_exp(const Tensor t);
Tensor tensor_log(const Tensor t);
Tensor tensor_softmax_all(const Tensor t);
Tensor tensor_softmax_axis(const Tensor t, i32 axis);
Tensor tensor_log_sum_exp_all(const Tensor t);
Tensor tensor_log_sum_exp_axis(const Tensor t, i32 axis);

Tensor tensor_scalar(f32 value);
bool tensor_fill(const Tensor t, f32 value);
Tensor tensor_reshape(const Tensor t, u32 ndims, const u32 *shape);
bool tensor_copy_data(Tensor dst, const Tensor src);

#endif
