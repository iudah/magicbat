#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "tensor_memory.h"
#include <stddef.h>
#include <stdint.h>

Tensor tensor_new(const u32 ndims, const u32 *shape) {
  if (ndims == 0 || shape == NULL)
    return NULL;

  Tensor t = tcalloc(1, sizeof(*t));
  if (t == NULL) {
    return NULL;
  }

  u32 *tmp = tcalloc(ndims, sizeof(u32));
  if (tmp == NULL) {
    tfree(t);
    return NULL;
  }

  u64 lenght = 1;
  for (u32 i = 0; i < ndims; i++) {
    lenght *= shape[i];
    tmp[i] = shape[i];
  }

  if (lenght == 0) {
    tfree(t);
    return NULL;
  }

  f32 *tmpdata = tcalloc(lenght, sizeof(f32));
  if (tmpdata == NULL) {
    tfree(t);
    tfree(tmp);
    return NULL;
  }

  t->ndims = ndims;
  t->shape = tmp;
  t->nelements = lenght;
  t->data = tmpdata;

  return t;
}