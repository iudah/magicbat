#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <stdint.h>

Tensor tensor_negate(const Tensor t) {
  if (!t)
    return NULL;

  Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
  if (res == NULL)
    return NULL;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto a = t->data->data[i];
    res->data->data[i] = -a;
  }

  return res;
}
