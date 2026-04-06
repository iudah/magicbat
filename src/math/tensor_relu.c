#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_relu(const Tensor t) {
  if (!t)
    return NULL;

  Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
  if (res == NULL)
    return NULL;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto a = t->data[i];
    res->data[i] = fmaxf(a, 0);
  }

  return res;
}

#define EPS (1e-8)
Tensor tensor_relu_backward(const Tensor t, const Tensor grad) {
  if (!t)
    return NULL;

  Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
  if (res == NULL)
    return NULL;

  auto nelement = tensor_num_elements(res);

  if (grad) {
    for (u32 i = 0; i < nelement; ++i) {
      res->data[i] = t->data[i] > EPS ? grad->data[i] : 0.0f;
    }
  } else {
    for (u32 i = 0; i < nelement; ++i) {
      auto a = t->data[i];
      res->data[i] = a > EPS ? 1.0f : 0.0f;
    }
  }

  return res;
}
