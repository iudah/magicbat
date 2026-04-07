#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "data_storage.h"
#include "tensor_memory.h"
#include <stddef.h>
#include <stdint.h>

Tensor tensor_new(const u32 ndims, const u32 *shape) {
  if (ndims == 0 || shape == NULL || ndims > MAX_DIMS)
    return NULL;

  Tensor t = tcalloc(1, sizeof(*t));
  if (t == NULL) {
    return NULL;
  }

  u32 *tmp = t->shape;

  u64 lenght = 1;
  for (u32 i = 0; i < ndims; i++) {
    lenght *= shape[i];
    tmp[i] = shape[i];
  }

  if (lenght == 0) {
    tfree(t);
    return NULL;
  }

  auto tmpdata = data_storage_new(lenght);
  if (tmpdata == NULL) {
    tfree(t);
    return NULL;
  }

  t->ndims = ndims;
  t->data = tmpdata;
  t->requires_grad = false;
  t->refcount = 1;
  t->is_tensor_type = true;

  return t;
}
