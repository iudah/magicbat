#include "tensor_prot.h"
#include "tensor.h"
#include "tensor_memory.h"
#include "tensor_odometer.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_max_axis(const Tensor tensor, i32 axis) {
  if (!tensor)
    return nullptr;

  if (axis < 0)
    axis += tensor->ndims;
  if (axis < 0)
    return nullptr;

  if (axis >= (i32)tensor->ndims)
    return nullptr;

  u32 *shape = tmalloc(sizeof(*shape) * tensor->ndims);
  for (u32 i = 0; i < tensor->ndims; ++i) {
    shape[i] = tensor->shape[i];
  }
  shape[axis] = 1;
  // guarantees zero
  Tensor res = tensor_new(tensor->ndims, shape);
  tfree(shape);
  if (!res)
    return nullptr;

  u32 outer_size = 1;
  u32 inner_size = 1;
  u32 axis_size = tensor->shape[axis];

  for (u32 i = 0; (i32)i < axis; ++i) {
    outer_size *= tensor->shape[i];
  }
  for (u32 i = axis + 1; i < tensor->ndims; ++i) {
    inner_size *= tensor->shape[i];
  }

  if (!tensor->is_contiguous) {
    u32 index[MAX_DIMS] = {0};

    for (u32 o_indx = 0; o_indx < outer_size; ++o_indx) {
      // u32 o_offset = o_indx * axis_size * inner_size;
      u32 r_offset = o_indx * inner_size;

      tensor_odometer_next(index, axis, tensor->shape);

      float *r_data = &res->data->data[r_offset];
      for (u32 i = 0; i < inner_size; ++i) {
        float max = -INFINITY;
        memset(index + axis + 1, 0, tensor->ndims - axis);
        for (u32 a_indx = 0; a_indx < axis_size; ++a_indx) {
          tensor_odometer_next(index + axis + 1, tensor->ndims - axis - 1,
                               tensor->shape + axis + 1);
          // u32 a_offset = o_offset + (a_indx * inner_size);
          // cache-miss read if inner_size > locality
          max = fmaxf(max, tensor_get(tensor, index));
        }
        r_data[i] = max;
      }
    }
  } else {
    for (u32 outer_indx = 0; outer_indx < outer_size; ++outer_indx) {
      u32 o_offset = outer_indx * axis_size * inner_size;
      u32 r_offset = outer_indx * inner_size;
      float *r_data = &res->data->data[r_offset];
      for (u32 i = 0; i < inner_size; ++i) {
        float max = -INFINITY;
        for (u32 axis_indx = 0; axis_indx < axis_size; ++axis_indx) {
          u32 a_offset = (o_offset + axis_indx) * inner_size;
          // cache-miss will occur if inner_size > locality (~64B)
          // But avoid ping-ponging res->data
          max = fmaxf(max, tensor->data->data[tensor->offset + a_offset + i]);
        }
        r_data[i] = max;
      }
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
