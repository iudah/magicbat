#ifndef TENSOR_NDIM_FLAT_INDEX_H
#define TENSOR_NDIM_FLAT_INDEX_H

#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/adt/type_alias.h"
#include "../../include/tensor.h"

static inline u32 tensor_ndim_flat_index(const Tensor tensor,
                                         const u32 *index) {
  TASSERT(tensor && "Null tensor");

  u32 flat = tensor->offset;
  for (u32 i = 0; i < tensor->ndims; ++i) {
    if (index[i] >= tensor->shape[i])
      return -1;
    flat += tensor->stride[i] * index[i];
  }

  return flat;
}

#endif
