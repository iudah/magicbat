#include "tensor.h"
#include "tensor_prot.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_tanh(const Tensor tensor) {
  TASSERT(tensor && "Null tensor");
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(tensor), tensor_shape(tensor));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val                         = tensor->data->data[tensor->offset + i];
    res->data->data[i + res->offset] = tanhf(val);
  }

  return res;
}


Tensor tensor_tanh_backward(const Tensor tensor, const Tensor grad) {
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(tensor), tensor_shape(tensor));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (grad) {
    for (u32 i = 0; i < nelement; ++i) {
      auto tanhx = tanhf(tensor->data->data[tensor->offset + i]);
      res->data->data[i + res->offset] =
          (1 - tanhx * tanhx) * grad->data->data[i + grad->offset];
    }
  } else {
    for (u32 i = 0; i < nelement; ++i) {
      auto val                         = tensor->data->data[i + tensor->offset];
      auto tanhx                       = tanhf(val);
      res->data->data[i + res->offset] = (1 - (tanhx * tanhx));
    }
  }

  return res;
}
