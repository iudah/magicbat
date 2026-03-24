#include "../../include/tensor/adt/tensor_prot.h"

bool tensor_set(Tensor t, u32 *index, f32 value) {
  if (tensor_index_out_of_bound(t, index))
    return false;

  u32 flat = 0;
  for (u32 i = 0; i < t->ndims; ++i) {
    flat = flat * t->shape[i] + index[i];
  }

  if (flat >= t->nelements)
    return false;

  return t->data[flat] = value;

  return true;
}
