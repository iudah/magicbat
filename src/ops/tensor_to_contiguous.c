#include "tensor.h"
#include "tensor_ndim_flat_index.h"
#include "tensor_odometer.h"
#include <stdatomic.h>
#include <stdint.h>

Tensor tensor_to_contiguous(Tensor tensor) {
  if (tensor->is_contiguous)
    return tensor;

  Tensor res = tensor_new(tensor->ndims, tensor->shape);
  auto index = tensor_odometer_new(tensor->ndims);
  for (u32 i = 0; i < res->data->nelements; ++i) {
    res->data->data[i] = tensor_get(tensor, index);
    tensor_odometer_next(index, res->ndims, tensor->shape);
  }
  tensor_destroy(index);
  return res;
}

void tensor_to_contiguous_inplace(Tensor tensor) {
  if (tensor->is_contiguous)
    return;

  u32 len = 1;
  for (u32 i = tensor->ndims; i-- > 0; --i) {
    len *= tensor->shape[i];
  }

#define N_BITS 64
  u32 n_packs = (len / N_BITS) + 1;
  u64 swapped[n_packs];
  for (u32 i = 0; i < n_packs; ++i)
    swapped[i] = 0;
#define SWAP(x) swapped[(x) / N_BITS] |= (1 << ((x) & (~N_BITS)))
#define IS_SWAPPED(x) (swapped[(x) / N_BITS] & (1 << ((x) & (~N_BITS))))

  auto index = tensor_odometer_new(tensor->ndims);
  for (u32 i = 0; i < tensor->data->nelements; ++i) {

    auto flat = tensor_ndim_flat_index(tensor, index);
    if (!IS_SWAPPED(i) && !IS_SWAPPED(flat)) {
      auto tmp = tensor->data->data[flat];
      tensor->data->data[flat] = tensor->data->data[i];
      tensor->data->data[i] = tmp;
      SWAP(i);
      SWAP(flat);
    }

    tensor_odometer_next(index, tensor->ndims, tensor->shape);
  }
  tensor_destroy(index);
}
