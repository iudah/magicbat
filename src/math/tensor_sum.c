#include "tensor_prot.h"
#include "tensor.h"
#include "tensor_memory.h"
#include "tensor_odometer.h"
#include "tensor_shapes_broadcast.h"
#include "tensor_shapes_equal.h"
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>

Tensor tensor_sum_axis(const Tensor tensor, i32 axis) {
  TASSERT(t && "Null tensor.");

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
        float sum = 0;
        memset(index + axis + 1, 0, (tensor->ndims - axis) * sizeof(u32));
        for (u32 a_indx = 0; a_indx < axis_size; ++a_indx) {
          tensor_odometer_next(index + axis + 1, tensor->ndims - axis - 1,
                               tensor->shape + axis + 1);
          // u32 a_offset = o_offset + (a_indx * inner_size);
          // cache-miss read if inner_size > locality
          sum += tensor_get(tensor, index);
        }
        r_data[i] = sum;
      }
    }
  } else {
    for (u32 o_indx = 0; o_indx < outer_size; ++o_indx) {
      u32 o_offset = o_indx * axis_size * inner_size;
      u32 r_offset = o_indx * inner_size;
      float *r_data = &res->data->data[r_offset];
      for (u32 i = 0; i < inner_size; ++i) {
        float sum = 0;
        for (u32 a_indx = 0; a_indx < axis_size; ++a_indx) {
          u32 a_offset = o_offset + (a_indx * inner_size);
          // cache-miss read if inner_size > locality
          sum += tensor->data->data[tensor->offset + a_offset + i];
        }
        r_data[i] = sum;
      }
    }
  }
  return res;
}

float tensor_sum_all(Tensor tensor) {
  TASSERT(t && "Null tensor.");

  if (!tensor)
    return -INFINITY;

  float sum = 0;
  u32 nelements = tensor_num_elements(tensor);
  for (u32 i = 0; i < nelements; ++i) {
    sum += tensor->data->data[i];
  }

  return sum;
}

Tensor tensor_sum_to_shape(Tensor tensor, u32 ndims, u32 *shape) {
  TASSERT(t && shape && "Null tensor or shape");

  if (tensor_shapes_equal_from_shape(tensor->ndims, tensor->shape, ndims,
                                     shape)) {
    atomic_fetch_add(&tensor->refcount, 1);
    return tensor;
  }

  u32 o_ndims = tensor->ndims > ndims ? tensor->ndims : ndims;
  u32 o_shape[MAX_DIMS] = {0};

  if (tensor_shapes_broadcast_from_shape(tensor->ndims, tensor->shape, ndims,
                                         shape, o_shape) &&
      tensor_shapes_equal_from_shape(tensor->ndims, tensor->shape, o_ndims,
                                     o_shape)) {

    // Guarantees a zero filled tensor
    Tensor res = tensor_new(ndims, shape);
    if (res == nullptr) {
      return nullptr;
    }

    u32 index[MAX_DIMS] = {0};
    do {
      u32 s_indx = 0;
      u32 t_indx = 0;

      for (u32 i = o_ndims, j = res->ndims; i-- > 0 && j-- > 0;) {
        s_indx += res->stride[j] * index[i];
      }
      for (u32 i = o_ndims, k = tensor->ndims; i-- > 0 && k-- > 0;) {
        t_indx += tensor->stride[k] * index[i];
      }

      res->data->data[res->offset + s_indx] +=
          tensor->data->data[tensor->offset + t_indx];
    } while (tensor_odometer_next(index, o_ndims, o_shape));
    return res;
  }
  return nullptr;
}
