#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_log(const Tensor t) {
  if (!t)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto a = t->data->data[i];
    res->data->data[i] = logf(a);
  }

  return res;
}
