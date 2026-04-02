#include "../../include/tensor/adt/tensor_prot.h"

bool tensor_set(Tensor t, u32 *index, f32 value) {
  u32 flat = 0;
  for (u32 i = 0; i < t->ndims; ++i) {
    if (index[i] >= t->shape[i])
      return false;
    flat = flat * t->shape[i] + index[i];
  }

  if (flat >= t->nelements)
    return false;

  t->data[flat] = value;

  return true;
}
