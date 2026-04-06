#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"

bool tensor_index_out_of_bound(const Tensor t, const u32 *index) {
  if (t == NULL || index == NULL)
    return false;

  for (u32 i = 0; i < t->ndims; ++i) {
    if (t->shape[i] <= index[i])
      return true;
  }

  return false;
}
