#ifndef TENSOR_H
#define TENSOR_H

#include "adt/type_alias.h"
#include <stdbool.h>

typedef struct tensor_struct *Tensor;

f32 tensor_get(Tensor t, u32 *index);
bool tensor_index_out_of_bound(Tensor t, u32 *index);
u32 tensor_ndims(Tensor t);
u32 tensor_num_elements(Tensor t);
bool tensor_set(Tensor t, u32 *index, f32 value);
const u32 *tensor_shape(Tensor t);

Tensor tensor_new(const u32 ndims, const u32 *shape);
bool tensor_destroy(Tensor t);

Tensor tensor_add(Tensor t, Tensor s);
Tensor tensor_sub(Tensor t, Tensor s);
Tensor tensor_mul(Tensor t, Tensor s);
Tensor tensor_div(Tensor t, Tensor s);
Tensor tensor_matmul(Tensor t, Tensor s);

bool tensor_fill(Tensor t, f32 value);

#endif
