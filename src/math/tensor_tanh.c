#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_tanh(const Tensor t) {
  TASSERT(t && "Null tensor");
  if (!t)
    return NULL;

  Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
  if (res == NULL)
    return NULL;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto a = t->data->data[i];
    res->data->data[i] = tanhf(a);
  }

  return res;
}

#define EPS (1e-8)
Tensor tensor_tanh_backward(const Tensor t, const Tensor grad) {
  if (!t)
    return NULL;

  Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
  if (res == NULL)
    return NULL;

  auto nelement = tensor_num_elements(res);

  if (grad) {
    for (u32 i = 0; i < nelement; ++i) {
      auto tanhx = tanhf(t->data->data[i]);
      res->data->data[i] = (1 - tanhx * tanhx) * grad->data->data[i];
    }
  } else {
    for (u32 i = 0; i < nelement; ++i) {
      auto a = t->data->data[i];
      auto tanhx = tanhf(a);
      res->data->data[i] = (1 - tanhx * tanhx);
    }
  }

  return res;
}
