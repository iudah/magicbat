#include "tensor_prot.h"
#include "tensor.h"

bool tensor_index_out_of_bound(const Tensor tensor, const u32 *index) {
  TASSERT(t && index && "Null tensor or index.");

  if (tensor == nullptr || index == nullptr)
    return false;

  for (u32 i = 0; i < tensor->ndims; ++i) {
    if (tensor->shape[i] <= index[i])
      return true;
  }

  return false;
}
