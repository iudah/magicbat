#include "tensor.h"
#include "tensor_odometer.h"
#include <stdatomic.h>
#include <stdint.h>

Tensor tensor_to_contiguous(Tensor tensor) {
  if (!tensor->is_contiguous)
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
