#ifndef TENSOR_H
#define TENSOR_H

#include "adt/type_alias.h"
#include <stdbool.h>

typedef struct tensor_struct *Tensor;

f32 tensor_get(const Tensor t, const u32 *index);
bool tensor_index_out_of_bound(const Tensor t, const u32 *index);
u32 tensor_ndims(const Tensor t);
u32 tensor_num_elements(const Tensor t);
bool tensor_set(const Tensor t, const u32 *index, f32 value);
const u32 *tensor_shape(const Tensor t);

Tensor tensor_new(const u32 ndims, const u32 *shape);
bool tensor_destroy(const Tensor t);

Tensor tensor_add(const Tensor t, const Tensor s);
Tensor tensor_sub(const Tensor t, const Tensor s);
Tensor tensor_mul(const Tensor t, const Tensor s);
Tensor tensor_div(const Tensor t, const Tensor s);
Tensor tensor_matmul(const Tensor t, const Tensor s);
Tensor tensor_sum_axis(const Tensor t, i32 axis);
f32 tensor_sum_all(const Tensor t);
Tensor tensor_max_axis(const Tensor t, i32 axis);
f32 tensor_max_all(const Tensor t);

Tensor tensor_relu(const Tensor t);
Tensor tensor_exp(const Tensor t);
Tensor tensor_softmax_all(const Tensor t);
Tensor tensor_softmax_axis(const Tensor t, i32 axis);

bool tensor_fill(const Tensor t, f32 value);
Tensor tensor_reshape(const Tensor t, u32 ndims, const u32 *shape);

#endif
