#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "../lifecycle/tensor_memory.h"
#include "tensor_odometer.h"
#include "tensor_shapes_broadcast.h"
#include "tensor_shapes_equal.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_sum_axis(const Tensor t, i32 axis) {
  if (!t)
    return NULL;

  if (axis < 0)
    axis += t->ndims;

  if (axis >= (i32)t->ndims)
    return NULL;

  u32 *shape = tmalloc(sizeof(*shape) * t->ndims);
  for (u32 i = 0; i < t->ndims; ++i) {
    shape[i] = t->shape[i];
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
    float *r_data = &res->data->data[r_offset];
    for (u32 i = 0; i < inner_size; ++i) {
      float sum = 0;
      for (u32 a = 0; a < axis_size; ++a) {
        u32 a_offset = o_offset + a * inner_size;
        // cache-miss read if inner_size > locality
        sum += t->data->data[a_offset + i];
      }
      r_data[i] = sum;
    }
  }
  return res;
}

float tensor_sum_all(Tensor t) {
  if (!t)
    return -INFINITY;

  float sum = 0;
  u32 nelements = tensor_num_elements(t);
  for (u32 i = 0; i < nelements; ++i) {
    sum += t->data->data[i];
  }

  return sum;
}

Tensor tensor_sum_to_shape(Tensor t, u32 ndims, u32 *shape) {

  u32 o_ndims = t->ndims > ndims ? t->ndims : ndims;
  u32 *t_stride = tmalloc(o_ndims * 4 * sizeof(u32));
  u32 *s_stride = t_stride + o_ndims;
  u32 *o_stride = s_stride + o_ndims;
  u32 *o_shape = o_stride + o_ndims;

  if (tensor_shapes_broadcast_from_shape(t->ndims, t->shape, ndims, shape,
                                         o_shape, o_stride, t_stride,
                                         s_stride) &&
      tensor_shapes_equal_from_shape(t->ndims, t->shape, o_ndims, o_stride)) {

    // Guarantees a zero filled tensor
    Tensor res = tensor_new(ndims, shape);
    if (res == NULL) {
      return NULL;
    }

    u32 *index = tensor_odometer_new(o_ndims);
    if (index == NULL) {
      tensor_destroy(res);
      res = NULL;
      return NULL;
    }
    do {
      u32 o_indx = 0;
      u32 t_indx = 0;

      for (u32 i = 0; i < o_ndims; ++i) {
        o_indx += o_stride[i] * index[i];
        t_indx += t_stride[i] * index[i];
      }

      res->data->data[o_indx] += t->data->data[t_indx];
    } while (tensor_odometer_next(index, o_ndims, o_shape));
    tensor_odometer_destroy(index);
    tfree(t_stride);
    return res;
  }
  tfree(t_stride);
  return NULL;
}
