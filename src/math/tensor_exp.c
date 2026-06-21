#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_exp(const Tensor tensor) {
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(tensor), tensor_shape(tensor));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val = tensor->data->data[i];
    res->data->data[i] = expf(val);
  }

  return res;
}
