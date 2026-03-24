#include "../../include/tensor/adt/tensor_prot.h"
#include "../../include/tensor/tensor.h"
#include "tensor_memory.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

bool tensor_destroy(Tensor t) {
  if (t == NULL)
    return false;

  if (t->data == NULL || t->nelements == 0 || t->ndims == 0 || t->shape == NULL)
    return false;

  tfree(t->data);
  tfree(t->shape);

  memset(t, 0, sizeof(*t));

  tfree(t);

  return true;
}