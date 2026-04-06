#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"

Tensor tensor_reshape(const Tensor t, u32 ndims, const u32 *shape) {
  if (!t)
    return NULL;

  Tensor res = tensor_new(ndims, shape);
  if (!res)
    return NULL;

  if (tensor_num_elements(res) != tensor_num_elements(t)) {
    tensor_destroy(res);
    return NULL;
  }

  // ToDo: Implement views
  u32 n = tensor_num_elements(t);
  for (u32 i = 0; i < n; ++i) {
    res->data[i] = t->data[i];
  }

  return res;
}