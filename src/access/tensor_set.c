#include "tensor_prot.h"
#include "tensor.h"
#include "tensor_ndim_flat_index.h"

bool tensor_contiguous_set(const Tensor tensor, const u32 *index, f32 value) {
  TASSERT(tensor && index && "Null tensor or index.");

  u32 flat = 0;
  for (u32 i = 0; i < tensor->ndims; ++i) {
    if (index[i] >= tensor->shape[i])
      return false;
    flat = (flat * tensor->shape[i]) + index[i];
  }

  if (flat >= tensor->data->nelements)
    return false;

  tensor->data->data[flat] = value;

  return true;
}

bool tensor_set(const Tensor tensor, const u32 *index, f32 value) {
  TASSERT(tensor && index && "Null tensor or index.");

  auto flat = tensor_ndim_flat_index(tensor, index);

  TASSERT(flat < tensor->data->nelements && "Index out of bound");

  if (flat >= tensor->data->nelements)
    return false;

  tensor->data->data[flat] = value;

  return true;
}
