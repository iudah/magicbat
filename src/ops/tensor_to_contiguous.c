#include "data_storage.h"
#include "tensor.h"
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
  for (u32 i = tensor->ndims; i-- > 0;) {
    len *= tensor->shape[i];
  }

  auto dest = data_storage_new(len);

  u32 index[MAX_DIMS] = {0};
  for (u32 i = 0; i < dest->nelements; ++i) {

    auto val = tensor_get(tensor, index);
    dest->data[i] = val;
    tensor_odometer_next(index, tensor->ndims, tensor->shape);
  }
  data_storage_destroy(tensor->data);
  tensor->data = dest;
  tensor->is_contiguous = true;
  tensor->offset = 0;
  for (u32 len = 1, i = tensor->ndims; i-- > 0;) {
    tensor->stride[i] = len;
    len *= tensor->shape[i];
  }
}
