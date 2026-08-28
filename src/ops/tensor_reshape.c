#include "tensor.h"
#include "tensor_odometer.h"
#include "tensor_prot.h"

Tensor tensor_reshape(const Tensor tensor, u32 ndims, const u32 *shape) {
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(ndims, shape);
  if (!res)
    return nullptr;

  if (tensor_num_elements(res) != tensor_num_elements(tensor)) {
    tensor_destroy(res);
    return nullptr;
  }

  u32 num = tensor_num_elements(tensor);
  if (tensor->is_contiguous) {
    for (u32 i = 0; i < num; ++i) {
      res->data->data[i] = tensor->data->data[i];
    }
  } else {
    u32 index[MAX_DIMS] = {0};
    for (u32 i = 0; i < num; ++i) {
      res->data->data[i] = tensor_get(tensor, index);
      tensor_odometer_next(index, tensor->ndims, tensor->shape);
    }
  }

  return res;
}
