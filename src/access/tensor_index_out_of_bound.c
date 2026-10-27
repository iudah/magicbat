#include "tensor.h"
#include "tensor_prot.h"

bool tensor_index_out_of_bound(const Tensor tensor, const u32 *index) {
  TASSERT(tensor && index && "Null tensor or index.");

  if (tensor == nullptr || index == nullptr)
    return false;

  for (u32 i = 0; i < tensor->ndims; ++i) {
    if (tensor->shape[i] <= index[i])
      return true;
  }

  return false;
}
