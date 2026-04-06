#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "../lifecycle/tensor_memory.h"
#include <stdint.h>

Tensor tensor_gather_axis(const Tensor t, const Tensor indices, i32 axis) {
  if (!t || !indices)
    return NULL;

  bool same_dim = false;
  if (t->ndims == indices->ndims)
    same_dim = true;
  else if (t->ndims != (indices->ndims + 1))
    return NULL;

  if (axis < 0)
    axis += t->ndims;

  if (axis >= (i32)t->ndims)
    return NULL;

  u32 *shape = tmalloc(sizeof(*shape) * t->ndims);
  for (u32 i = 0; i < t->ndims; ++i) {
    shape[i] = t->shape[i];
    if (shape[i] !=
        indices->shape[i + ((!same_dim && i >= (u32)axis) ? 1 : 0)]) {
      tfree(shape);
      return NULL;
    }
  }
  shape[axis] = 1;
  // guarantees zero
  Tensor res = tensor_new(t->ndims, shape);
  tfree(shape);
  if (!res)
    return NULL;

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
    float *r_data = &res->data[r_offset];
    float *i_data = &indices->data[r_offset];
    // for (u32 a = 0; a < axis_size; ++a)
    {
      for (u32 i = 0; i < inner_size; ++i) {
        // cache-miss will occur if inner_size > locality (~64B)
        // But avoid ping-ponging res->data
        u32 cls = (u32)i_data[i];
        u32 a_offset = o_offset + cls * inner_size + i;
        r_data[i] = t->data[a_offset];
      }
    }
  }
  return res;
}
