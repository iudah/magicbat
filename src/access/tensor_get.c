#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <math.h>

f32 tensor_get(const Tensor t, const u32 *index) {
  u32 flat = 0;
  for (u32 i = 0; i < t->ndims; ++i) {
    if (index[i] >= t->shape[i])
      return -INFINITY;
    flat = flat * t->shape[i] + index[i];
  }

  if (flat >= t->nelements)
    return -INFINITY;

  return t->data[flat];
}
