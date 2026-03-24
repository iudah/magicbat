#include "../../include/tensor/adt/tensor_prot.h"
#include <math.h>

f32 tensor_get(Tensor t, u32 *index) {
  if (tensor_index_out_of_bound(t, index))
    return -INFINITY;

  u32 flat = 0;
  for (u32 i = 0; i < t->ndims; ++i) {
    flat = flat * t->shape[i] + index[i];
  }

  if (flat >= t->nelements)
    return -INFINITY;

  return t->data[flat];
}
