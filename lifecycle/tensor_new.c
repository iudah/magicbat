#include "../adt/tensor_prot.h"
#include "../tensor.h"
#include "tensor_memory.h"
#include <stddef.h>
#include <stdint.h>

Tensor tensor_new(u32 ndims, u32 *shape) {
  if (ndims == 0 || shape == NULL)
    return NULL;

  Tensor t = tcalloc(1, sizeof(*t));
  if (t == NULL) {
    return NULL;
  }

  mem tmp = tcalloc(ndims, sizeof(u32));

  u64 lenght = 1;
  for (u32 i = 0; i < ndims; i++)
    lenght *= shape[i];

  if (tmp == NULL || lenght == 0) {
    tfree(t);
    return NULL;
  }

  t->ndims = ndims;
  t->shape = tmp;
  t->length = lenght;

  return t;
}