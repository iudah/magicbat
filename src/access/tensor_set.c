#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"

bool tensor_set(const Tensor t, const u32 *index, f32 value) {
  TASSERT(t && index && "Null tensor or index.");

  u32 flat = 0;
  for (u32 i = 0; i < t->ndims; ++i) {
    if (index[i] >= t->shape[i])
      return false;
    flat = flat * t->shape[i] + index[i];
  }

  if (flat >= t->data->nelements)
    return false;

  t->data->data[flat] = value;

  return true;
}
