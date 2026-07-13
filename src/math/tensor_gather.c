#include "tensor.h"
#include "tensor_prot.h"
#include <stdint.h>

Tensor tensor_gather_axis(const Tensor tensor, const Tensor indices, i32 axis) {
  if (!tensor || !indices)
    return nullptr;

  if (tensor->ndims - indices->ndims > 1)
    return nullptr;

  if (axis < 0)
    axis += tensor->ndims;

  if (axis >= (i32)tensor->ndims)
    return nullptr;

  u32 shape[MAX_DIMS] = {0};
  for (u32 i = 0, j = 0; i < tensor->ndims; ++i) {
    shape[i] = tensor->shape[i];
    if (j < indices->ndims && shape[i] != indices->shape[j])
      return nullptr;
    ++j;
    if (i == (u32)axis)
      --j;
  }
  shape[axis] = 1;
  // guarantees zero
  Tensor res = tensor_new(tensor->ndims, shape);
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

  if (inner_size == 1)
    for (u32 o_index = 0; o_index < outer_size; ++o_index) {
      u32 o_offset = o_index * axis_size * inner_size;
      u32 r_offset = o_index * inner_size;
      float *r_data = &res->data->data[r_offset];
      float *i_data = &indices->data->data[r_offset];
      // for (u32 a = 0; a < axis_size; ++a)
      {
        {
          // cache-miss will occur if inner_size > locality (~64B)
          // But avoid ping-ponging res->data
          u32 cls = (u32)i_data[0];
          u32 a_offset = o_offset + (cls * inner_size);
          r_data[0] = tensor->data->data[a_offset];
        }
      }
    }
  else
    for (u32 o_index = 0; o_index < outer_size; ++o_index) {
      u32 o_offset = o_index * axis_size * inner_size;
      u32 r_offset = o_index * inner_size;
      float *r_data = &res->data->data[r_offset];
      float *i_data = &indices->data->data[r_offset];
      // for (u32 a = 0; a < axis_size; ++a)
      {
        for (u32 i = 0; i < inner_size; ++i) {
          // cache-miss will occur if inner_size > locality (~64B)
          // But avoid ping-ponging res->data
          u32 cls = (u32)i_data[i];
          u32 a_offset = o_offset + (cls * inner_size) + i;
          r_data[i] = tensor->data->data[a_offset];
        }
      }
    }
  return res;
}
