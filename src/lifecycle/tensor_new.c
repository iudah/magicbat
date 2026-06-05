#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "data_storage.h"
#include "tensor_memory.h"
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

Tensor tensor_new(const u32 ndims, const u32 *shape) {
  if (ndims == 0 || shape == nullptr || ndims > MAX_DIMS)
    return nullptr;

  Tensor tensor = tcalloc(1, sizeof(*tensor));
  if (tensor == nullptr) {
    return nullptr;
  }

  u32 *tmp = tensor->shape;

  u64 lenght = 1;
  for (u32 i = ndims; i-- > 0;) {
    tensor->stride[i] = lenght;
    lenght *= shape[i];
    tmp[i] = shape[i];
  }

  if (lenght == 0) {
    tfree(tensor);
    return nullptr;
  }

  auto tmpdata = data_storage_new(lenght);
  if (tmpdata == nullptr) {
    tfree(tensor);
    return nullptr;
  }

  tensor->ndims = ndims;
  tensor->data = tmpdata;
  tensor->requires_grad = false;
  tensor->refcount = 1;
  tensor->is_tensor_type = true;
  tensor->is_contiguous = true;

  return tensor;
}

Tensor tensor_view(const Tensor src) {

  Tensor tensor = tcalloc(1, sizeof(*tensor));
  if (tensor == nullptr) {
    return nullptr;
  }

  *tensor = *src;

  atomic_fetch_add(&tensor->refcount, 1);
  atomic_fetch_add(&tensor->data->refcount, 1);
  tensor->is_tensor_type = true;
  tensor->requires_grad = false;
  tensor->is_contiguous = false;

  return tensor;
}
