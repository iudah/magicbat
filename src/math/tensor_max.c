#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "../lifecycle/tensor_memory.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_max_axis(const Tensor t, i32 axis) {
  if (!t)
    return nullptr;

  if (axis < 0)
    axis += t->ndims;

  if (axis >= (i32)t->ndims)
    return nullptr;

  u32 *shape = tmalloc(sizeof(*shape) * t->ndims);
  for (u32 i = 0; i < t->ndims; ++i) {
    shape[i] = t->shape[i];
  }
  shape[axis] = 1;
  // guarantees zero
  Tensor res = tensor_new(t->ndims, shape);
  tfree(shape);
  if (!res)
    return nullptr;

  u32 outer_size = 1;
  u32 inner_size = 1;
  u32 axis_size = t->shape[axis];

  for (u32 i = 0; (i32)i < axis; ++i) {
    outer_size *= t->shape[i];
  }
  for (u32 i = axis + 1; i < t->ndims; ++i) {
    inner_size *= t->shape[i];
  }

  for (u32 o = 0; o < outer_size; ++o) {
    u32 o_offset = o * axis_size * inner_size;
    u32 r_offset = o * inner_size;
    float *r_data = &res->data->data[r_offset];
    for (u32 i = 0; i < inner_size; ++i) {
      float max = -INFINITY;
      for (u32 a = 0; a < axis_size; ++a) {
        u32 a_offset = o_offset + a * inner_size;
        // cache-miss will occur if inner_size > locality (~64B)
        // But avoid ping-ponging res->data
        max = fmaxf(max, t->data->data[a_offset + i]);
      }
      r_data[i] = max;
    }
  }
  return res;
}

float tensor_max_all(const Tensor tensor) {
  if (!tensor)
    return -NAN;

  float max = -INFINITY;
  u32 nelements = tensor_num_elements(tensor);
  for (u32 i = 0; i < nelements; ++i) {
    max = fmaxf(max, tensor->data->data[i]);
  }

  return max;
}
