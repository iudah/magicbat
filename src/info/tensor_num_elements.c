#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "type_alias.h"

u32 tensor_num_elements(Tensor tensor) {
  if (tensor->is_contiguous)
    return tensor->shape[0] * tensor->stride[0];

  u32 len = 1;
  for (u32 i = 0; i < tensor->ndims; ++i)
    len *= tensor->shape[i];
  return len;
}
