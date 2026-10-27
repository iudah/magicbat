#include "tensor.h"
#include "tensor_prot.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_log(const Tensor tensor) {
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(tensor), tensor_shape(tensor));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val                         = tensor->data->data[i + tensor->offset];
    res->data->data[res->offset + i] = logf(val);
  }

  return res;
}
