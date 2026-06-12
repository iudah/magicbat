#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"

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
  for (u32 i = 0; i < num; ++i) {
    res->data[i] = tensor->data[i];
  }

  return res;
}
