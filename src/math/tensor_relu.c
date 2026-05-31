#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_relu(const Tensor t) {
  TASSERT(t && "Null tensor");
  if (!t)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto a = t->data->data[i];
    res->data->data[i] = fmaxf(a, 0);
  }

  return res;
}

#define EPS (1e-8)
Tensor tensor_relu_backward(const Tensor t, const Tensor grad) {
  if (!t)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (grad) {
    for (u32 i = 0; i < nelement; ++i) {
      res->data->data[i] = t->data->data[i] > EPS ? grad->data->data[i] : 0.0f;
    }
  } else {
    for (u32 i = 0; i < nelement; ++i) {
      auto a = t->data->data[i];
      res->data->data[i] = a > EPS ? 1.0f : 0.0f;
    }
  }

  return res;
}
