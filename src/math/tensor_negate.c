#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <stdint.h>

Tensor tensor_negate(const Tensor tensor) {
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(tensor), tensor_shape(tensor));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val = tensor->data->data[i];
    res->data->data[i] = -val;
  }

  return res;
}

void tensor_negate_inplace(Tensor restrict tensor) {
  if (!tensor)
    return;

  Tensor res = tensor;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val = tensor->data->data[i];
    res->data->data[i] = -val;
  }
}
