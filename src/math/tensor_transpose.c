#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <stdint.h>

Tensor tensor_transpose(const Tensor t) {
  TASSERT(t && "Null tensor.");

  if (!t)
    return nullptr;
  if (t->ndims != 2)
    return nullptr;

  u32 row = t->shape[0];
  u32 col = t->shape[1];

  // guarantees zeros
  Tensor res = tensor_new(2, (u32[]){col, row});
  if (res == nullptr)
    return nullptr;

  for (u32 i = 0; i < row; ++i) {
    float *t_data = &t->data->data[i * col];
    for (u32 j = 0; j < col; ++j) {
      float t_val = t_data[j];
      res->data->data[j * row + i] = t_val;
    }
  }

  return res;
}
