#include "tensor.h"
#include "tensor_odometer.h"
#include "tensor_prot.h"
#include "type_alias.h"
#include <string.h>
#include <strings.h>

void concat_non_contiguous(Tensor concat, u32 axis, Tensor *tensors,
                           const u32 *bounds, const u32 n_tensor) {
  u32 pivot_index = 0;
  u32 pivot = 0;
  u32 pivot_tensor = 0;

  auto ndims = concat->ndims;
  auto data = concat->data->data;
  u32 index[MAX_DIMS] = {0};
  // for (u32 i = 0; i < concat->data->nelements; ++i) {
  /*if (axis == 0 && tensors[pivot_tensor]->is_contiguous) {
    auto nelements = tensors[pivot_tensor]->data->nelements;
    memcpy(data, tensors[pivot_tensor]->data->data, nelements * sizeof(f32));
    data += nelements;
  } else*/
  // {
  // tensor_odometer_reset(ndims, index);
  do {
    pivot = index[axis];
    for (pivot_tensor = 0; pivot_tensor < n_tensor; ++pivot_tensor) {
      if (pivot < bounds[pivot_tensor]) {
        pivot_index = pivot - (pivot_tensor > 0 ? bounds[pivot_tensor - 1] : 0);
        break;
      }
    }

    index[axis] = pivot_index;

    *data = tensor_get(tensors[pivot_tensor], index);
    ++data;

    index[axis] = pivot;

  } while (tensor_odometer_next(index, ndims, concat->shape));
  // }
  //}
}

Tensor tensor_concat(u32 n_tensor, Tensor *tensors, u32 axis) {
  if (n_tensor == 1)
    return *tensors;

  u32 shape[MAX_DIMS];
  u32 ndims = tensors[0]->ndims;
  bool all_contiguous = true;
  for (u32 i = 0; i < ndims; ++i) {
    shape[i] = tensors[0]->shape[i];
  }

  u32 bounds[n_tensor];
  bounds[0] = tensors[0]->shape[axis];

  for (u32 i = 1; i < n_tensor; ++i) {
    all_contiguous = all_contiguous && tensors[i]->is_contiguous;
    shape[axis] += tensors[i]->shape[axis];
    bounds[i] = shape[axis];

    if (i == axis)
      continue;

    if (tensors[i]->ndims != ndims)
      return nullptr;

    for (u32 j = 0; j < ndims; ++j) {
      if (tensors[i]->shape[j] != shape[j])
        return nullptr;
    }
  }

  u32 offset = 0;
  auto concat = tensor_new(ndims, shape);
  if (axis == 0 && all_contiguous) {
    auto data = concat->data->data;
    for (u32 i = 0; i < n_tensor; ++i) {
      memcpy(data + offset, tensors[i]->data->data,
             tensors[i]->data->nelements * sizeof(f32));
      offset += tensors[i]->data->nelements;
    }
  } else {
    concat_non_contiguous(concat, axis, tensors, bounds, n_tensor);
  }

  return concat;
}
